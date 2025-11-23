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
        context = torch.empty(0, dtype=torch.long).to("cuda")
        action_mask = torch.empty(0, dtype=torch.long).to("cuda")

        #system_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        #print(system_text)
        #system_text_ids = self.tokenizer(system_text, return_tensors="pt").to("cuda")
        #context_tensors.append(system_text_ids.input_ids)
        messages = [
            { "role": "system", "content": "You are playing Blackjack. At each step, you'll be given the state of the game. Respond with HIT or STAY. Do not add commentary." },
        ]

        while not game.get_state()["game_over"]:
            visible_state = {
                "your_hand": game.get_state()["player_hand"],
                "dealer_hand": ["Hidden"] + game.get_state()["dealer_hand"][1:]
            }
            state_str = str(visible_state)
            messages.append({ "role": "user", "content": state_str })

            messages_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            messages_encoding = self.tokenizer(messages_text, return_tensors="pt").to("cuda")
            context = torch.cat((context, messages_encoding["input_ids"]), dim=1)
            action_mask = torch.cat((action_mask, torch.zeros((1, len(messages_encoding["input_ids"][0])), dtype=torch.long).to("cuda")), dim=1)
            print("context:", context)
            print("action_mask:", action_mask)

            with torch.no_grad():
                outputs = self.model.generate(
                    **messages_encoding,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=1.,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )
            print(outputs)

            # Combine up response tokens and update the action mask.
            response_ids = outputs.sequences[0, messages_encoding.input_ids.shape[1]:]
            context = torch.cat((context, response_ids.reshape((1, len(response_ids))) ), dim=1)
            print(context)

            cur_action_mask = torch.ones(response_ids.shape, dtype=torch.long).to("cuda")
            cur_action_mask[response_ids == self.tokenizer.eos_token_id] = 0
            cur_action_mask = torch.reshape(cur_action_mask, (1, len(cur_action_mask)))
            action_mask = torch.cat((action_mask, cur_action_mask), dim=1)
            print(action_mask)

            # decode and take the action, gather an observation
            action = self.tokenizer.decode(response_ids[:-1])
            print(action)
            if "HIT" == action:
                observation = game.hit()
            elif "STAY" == action:
                observation = game.stand()
            else:
                raise ValueError("bad action")

            print(observation)

            raise NotImplementedError()
            

#            current_context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#            #print(state_str)
#            inputs = self.tokenizer(current_context, return_tensors="pt").to("cuda")
#            print(inputs)
#
#            with torch.no_grad():
#                outputs = self.model.generate(
#                    **inputs,
#                    max_new_tokens=32,
#                    do_sample=True,
#                    temperature=1.,
#                    pad_token_id=self.tokenizer.pad_token_id,
#                    return_dict_in_generate=True,
#                )
#
#            new_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]
#
#            response_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
#            print(response_text)
#
#            messages.append({ "role": "assistant", "content": response_text })
#
#            if "HIT" == response_text:
#                observation = game.hit()
#            elif "STAY" == response_text:
#                observation = game.stand()
#            else:
#                #print("invalid action")
#                break

            #print(observation)

        #full_context = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        return {
            "game": game,
            "full_context": messages,
        }

    def compute_reward(self, trajectory):
        total_reward = 0.
        if not trajectory["game"].get_state()["game_over"]:
            # agent did not play properly
            print("game did not finish")
            return -2.
        total_reward += 0.5
        outcome = trajectory["game"]._determine_winner()
        if "push" in outcome.lower():
            # tie
            print("tie")
            total_reward += 0.5
        elif "player wins" in outcome.lower():
            # agent win
            print("agent win")
            total_reward += 2.
        else:
            print("agent loss")
            # agent loss
            return -1.
        return total_reward


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

            # Score results
            scores = [self.compute_reward(t) for t in trajectories]
            rewards = [] 
            for i, t in enumerate(trajectories):
                context = t["full_context"]
                tmp = torch.zeros(len(self.tokenizer.apply_chat_template(context, tokenize=True, padding="max_length", max_length=512)))
                mask = action_mask(self.tokenizer, context)
                tmp[mask == 1] = scores[i]
                rewards.append(tmp)
            rewards = torch.stack(rewards, dim=0)
            #print(rewards)

            
            if len(set(rewards)) == 1:
                # no signal for training since the scores were the same
                continue

            # compute loss

def action_mask(tokenizer, context):
    tokens = tokenizer.apply_chat_template(context, tokenize=True, padding="max_length", max_length=512)
    target_words = ["HIT", "STAY"]
    target_ids = tokenizer(target_words)["input_ids"]
    action_indices = []
    lengths = []
    for targets in target_ids:
        for i, _ in enumerate(tokens):
            found = True
            for j, tok2 in enumerate(targets):
                if tok2 != tokens[i + j]:
                    found = False
                    break
            if found:
                action_indices.append(i)
                lengths.append(len(targets))

    mask = torch.zeros(len(tokens), dtype=torch.long)

    for i, index in enumerate(action_indices):
        run_len = lengths[i]
        mask[index : index + run_len] = 1

    return mask 



def main():
    trainer = SimpleGRPOTrainer("ibm-granite/granite-4.0-350m")
    tokenizer = trainer.tokenizer

    games = [blackjack.Blackjack() for _ in range(4)]
    games[0].start_round()
    trainer.generate_trajectory(games[0])


if "__main__" == __name__:
    main()

