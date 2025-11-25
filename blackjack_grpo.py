import copy

import torch
import torch.nn.functional as F

import transformers

import blackjack


class SimpleGRPOTrainer:
    def __init__(self, model_id):
        # Tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Policy Model
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

    def generate_trajectory(self, game):
        action_mask = None

        messages = [
            { "role": "system", "content": "You are playing Blackjack. At each step, you'll be given the state of the game. Respond with HIT or STAY. Do not add commentary." },
        ]

        invalid_action = False
        while not game.get_state()["game_over"]:
            visible_state = {
                "your_hand": game.get_state()["player_hand"],
                "dealer_hand": ["Hidden"] + game.get_state()["dealer_hand"][1:]
            }
            state_str = str(visible_state)
            #print(state_str)
            messages.append({ "role": "user", "content": state_str })

            messages_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            messages_encoding = self.tokenizer(messages_text, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = self.model.generate(
                    **messages_encoding,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=1.,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
            #print(outputs)

            # Combine up response tokens and update the action mask.
            response_ids = outputs.sequences[0, messages_encoding.input_ids.shape[1]:]
            if action_mask is None:
                action_mask = torch.zeros(outputs.sequences.shape[1])
            else:
                new_ids = outputs.sequences[0, action_mask.shape[0]:]
                action_mask = torch.cat((action_mask, torch.zeros(new_ids.shape[0])), dim=0)
            action_mask[messages_encoding.input_ids.shape[1]:][response_ids != self.tokenizer.eos_token_id] = 1
            print(outputs.sequences)
            print(action_mask)
            
            # decode and take the action, gather an observation
            action = self.tokenizer.decode(response_ids[:-1])
            # Update messages for the next iteration
            messages.append({ "role": "assistant", "content": action })

            #print(action)
            if "HIT" == action:
                observation = game.hit()
            elif "STAY" == action:
                observation = game.stand()
            else:
                invalid_action = True
                break

        # calculate reward for trajectory
        trajectory = { "game": game, "messages": messages }
        reward = self.compute_reward(trajectory)
        reward_tensor = torch.zeros_like(action_mask, dtype=torch.float)
        reward_tensor[torch.where(action_mask == 1)] = reward
        trajectory["reward_tensor"] = reward_tensor

        # compute logprobs for trajectory
        trajectory_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False).strip()
        print(trajectory_text)
        trajectory_encoding = self.tokenizer(trajectory_text, return_tensors="pt").to("cuda")
        print(trajectory_encoding.input_ids)
        print(trajectory_encoding.input_ids.shape)
        print(reward_tensor.shape)
        assert trajectory_encoding.input_ids.shape[1] == reward_tensor.shape[0]

        return trajectory

    def compute_reward(self, trajectory):
        total_reward = 0.
        if not trajectory["game"].get_state()["game_over"]:
            # agent did not play properly
            #print("game did not finish")
            return -2.
        total_reward += 0.5
        outcome = trajectory["game"]._determine_winner()
        if "push" in outcome.lower():
            # tie
            #print("tie")
            total_reward += 0.5
        elif "player wins" in outcome.lower():
            # agent win
            #print("agent win")
            total_reward += 2.
        else:
            #print("agent loss")
            # agent loss
            return -1.
        return total_reward

    def grpo_loss(self, trajectories):
        reward_tensor = torch.stack([traj["reward_tensor"] for traj in trajectories], dim=0)
        print(reward_tensor)
        print(reward_tensor.shape)

    def train_step(self, games):
        self.model.train()
        total_loss = 0.

        for game in games:
            game.start_round()

            trajectories = [
                self.generate_trajectory(copy.deepcopy(game))
                for _ in range(4) # num generations
            ]
            #print(trajectories)

            #if len(set(rewards)) == 1:
                # no signal for training since the scores were the same
            #    continue

            # compute loss
            loss = self.grpo_loss(trajectories)


def main():
    trainer = SimpleGRPOTrainer("ibm-granite/granite-4.0-350m")
    tokenizer = trainer.tokenizer

    games = [blackjack.Blackjack() for _ in range(100)]

    #trainer.train_step(games)
    games[0].start_round()
    trainer.generate_trajectory(games[0])


if "__main__" == __name__:
    main()

