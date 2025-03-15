import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Config,
    T5Tokenizer,
    mpu)
from typing import Optional


class Reward():
    def __init__(self, args, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        #print("###max_length in reward###",args.max_length)

    def get_input_batch(self, input_ids, gen_ids):
        attention_mask = (input_ids != self.pad_token_id)

        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": gen_ids.contiguous(),
            "use_cache": False
        }

        #print("labael in get_input_batch", model_inputs["labels"].size(), type(model_inputs["labels"]))
        return model_inputs
    
    

    def reward_fn(self, input_ids: torch.Tensor, gen_ids: torch.Tensor, 
                  inf_mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:

        model_inputs = self.get_input_batch(input_ids, gen_ids)
        # torch.set_printoptions(profile="full")
        # print("###labels in rewardfnn###", model_inputs["labels"])
        # torch.set_printoptions(profile="default") # reset
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**model_inputs)
        
        logits = outputs.logits  # [batch, seq_len, vocab_size]
        
        # Normalize logits by subtracting mean
        logits = logits - torch.mean(logits, dim=-1, keepdim=True)
        
        # Create mask for valid decoder positions (non-padding)
        mask = (gen_ids != self.pad_token_id)
        
        # Apply mask to zero out logits for padding positions
        logits = logits * mask.unsqueeze(-1)
        
        # Get logits for the tokens that were actually generated
        selection_value = torch.gather(logits, -1, gen_ids.unsqueeze(-1)).squeeze(-1)
        
        # Compute log sum exp over vocabulary for normalization
        next_state_value = torch.logsumexp(logits, dim=-1)
        
        # Apply mask to zero out scores for padding positions
        next_state_value = next_state_value * mask
        
        # Compute log probabilities
        scores = selection_value - next_state_value
        
        # Verify no inf/nan values
        assert all((~torch.isinf(scores.view(-1))) & (~torch.isnan(scores.view(-1))))
        
        # Verify shapes match
        assert scores.size() == gen_ids.size()
        
        return {
            "rewards": scores,
            "inf_mask": inf_mask
        }




    # Old function+
    # def reward_fn(self, input_ids, gen_ids, inf_mask=None, output_pos=True):
    #     """
    #     Teacher forcing computing KL divergence.
    #     Rewards are calculated as the difference between the log-probability of the chosen token and the log-sum-exp of all possible next tokens.

    #     input_ids: Model input tokens
    #     Gen_ids: Generated Model output Tokens
    #     inf_mask: positions where the logits for the outputs are inf
    #     """
    #     # not include eos token

    #     # torch.set_printoptions(profile="full")
    #     # print("inputids",input_ids)
    #     # print("genids",gen_ids)
    #     # torch.set_printoptions(profile="default") # reset
    #     # print("shape inputids",input_ids.size())
        

        
    #     self.model.eval()
    #     # input_ids = input_ids.repeat(1, 1)
    #     model_inputs = self.get_input_batch(input_ids, gen_ids)#output_pos=output_pos)

    #     print("###model_inputs in rewardfn###", model_inputs)

    #     torch.set_printoptions(profile="full")
    #     print("labels:", model_inputs["labels"]) # prints the whole tensor
    #     torch.set_printoptions(profile="default") # reset
        
    #     # Labels are student output
    #     with torch.no_grad():
    #         outputs = self.model(**model_inputs)

    #     logits = outputs.logits.contiguous() # (B, L, V)
    #     # Normalize Logits
    #     if self.args.model_parallel:
    #         logits = logits - mpu.parallel_mean(logits.float(), dim=-1).unsqueeze(-1)
    #     else:
    #         logits = logits - torch.mean(logits, dim=-1, keepdim=True)
        


    #     mask = model_inputs["attention_mask"]
    #     print(f"### shap logits before slicing in rewardfn###", logits.size())
    #     print(f"### shap mask before slicing in rewardfn###", mask.size())

    #     # Apply mask (added dummy dimension) to logits
    #     logits = logits * mask.unsqueeze(-1)
        
    #     logits = logits[:, :input_ids.size(-1), :].contiguous()
    #     mask = mask[:, :input_ids.size(-1)].contiguous()

    #     print(f"### shap logits in rewardfn###", logits.size())
    #     print(f"### shap mask in rewardfn###", mask.size())


    #     if self.args.model_parallel:
    #         selection_value = mpu.parallel_gather(logits[:, :, :], -1, model_inputs["input_ids"][:, input_ids.size(-1):, None]).squeeze(-1)
    #     else:
    #         # Represents logits for gen_ids, slices from input_ids on
    #         selection_value = torch.gather(logits, -1, model_inputs["input_ids"][:, input_ids.size(-1):, None]).squeeze(-1)
    #         print(f"###selection_value in rewardfn###", selection_value)

    #     #current_logits = logits.contiguous()
    #     if self.args.model_parallel:
    #         next_state_value = mpu.parallel_logsumexp(logits.float(), dim=-1)
    #     else:
    #         next_state_value = torch.logsumexp(logits, dim=-1)
    #         print(f"###next_state_value in rewardfn###", next_state_value)
    #     next_state_value = next_state_value * mask[:, :]
    #     #raw_next_state_value = next_state_value

        
    #     scores = selection_value - next_state_value
    #     print(f"###scores in rewardfn###", scores)
    #     assert all((~torch.isinf(scores.view(-1))) & (~torch.isnan(scores.view(-1))))
        
    #     #thrown
    #     #assert scores.size() == gen_ids.size()
        
    #     return {
    #         "rewards": scores,
    #         "inf_mask": inf_mask
    #     }
    
    # def reward_fn(self, input_ids, gen_ids, inf_mask=None, output_pos=True):
    #     """
    #     Teacher forcing computing KL divergence with added debugging.
        
    #     Args:
    #         input_ids: Model input tokens
    #         gen_ids: Generated Model output tokens
    #         inf_mask: positions where the logits for the outputs are inf
    #         output_pos: Whether to output position information
            
    #     Returns:
    #         Dict containing rewards and inf_mask
    #     """
    #     self.model.eval()
        
    #     # Debug input shapes
    #     print(f"Input shapes:")
    #     print(f"input_ids shape: {input_ids.shape}")
    #     print("content input", input_ids)
    #     print(f"gen_ids shape: {gen_ids.shape}")
        
    #     # Get model inputs and verify
    #     model_inputs = self.get_input_batch(input_ids, gen_ids)
    #     print(f"Model inputs shapes:")
    #     # for k, mask, v in model_inputs.items():
    #     #     print("lookhere, input:",k, "\n mask:", mask, "\n gen_ids:", v)
    #     #     print(f"{k} shape: {v.shape}")
        
    #     with torch.no_grad():
    #         outputs = self.model(**model_inputs)
    #         logits = outputs.logits.contiguous()  # (B, L, V)
    #         print(f"Initial logits shape: {logits.shape}")
            
    #         # Normalize logits
    #         if self.args.model_parallel:
    #             logits = logits - mpu.parallel_mean(logits.float(), dim=-1).unsqueeze(-1)
    #         else:
    #             logits = logits - torch.mean(logits, dim=-1, keepdim=True)
            
    #         mask = model_inputs["attention_mask"]
    #         print(f"Mask shape: {mask.shape}")
            
    #         # Add dummy dimenision for multplication
    #         logits = logits * mask.unsqueeze(-1)
    #         # Apply mask and slice
    #         slice_start = input_ids.size(-1)
    #         print(f"Slice start: {slice_start}")
    #         logits = logits[:, slice_start:, :].contiguous()
    #         mask = mask[:, slice_start:].contiguous()
            
    #         print(f"After slicing:")
    #         print(f"logits shape: {logits.shape}")
    #         print(f"mask shape: {mask.shape}")
            
    #         # Compute selection value
    #         if self.args.model_parallel:
    #             selection_value = mpu.parallel_gather(
    #                 logits[:, :-1, :], 
    #                 -1, 
    #                 model_inputs["input_ids"][:, input_ids.size(-1):, None]
    #             ).squeeze(-1)
    #         else:
    #             gather_indices = model_inputs["input_ids"][:, input_ids.size(-1):, None]
    #             print(f"Gather indices shape: {gather_indices.shape}")
    #             print(f"Logits for gather shape: {logits[:, :-1, :].shape}")
    #             selection_value = torch.gather(
    #                 logits[:, :-1, :], 
    #                 -1, 
    #                 gather_indices
    #             ).squeeze(-1)
            
    #         print(f"Selection value shape: {selection_value.shape}")
            
    #         # Compute next state value
    #         current_logits = logits[:, :-1, :].contiguous()
    #         if self.args.model_parallel:
    #             next_state_value = mpu.parallel_logsumexp(current_logits.float(), dim=-1)
    #         else:
    #             next_state_value = torch.logsumexp(current_logits, dim=-1)
            
    #         next_state_value = next_state_value * mask[:, :-1]
    #         print(f"Next state value shape: {next_state_value.shape}")
            
    #         # Compute final scores
    #         scores = selection_value - next_state_value
    #         print(f"Final scores shape: {scores.shape}")
            
    #         # Validation checks
    #         if scores.numel() == 0:
    #             print("WARNING: Scores tensor is empty!")
    #             print(f"Selection value stats: min={selection_value.min() if selection_value.numel() > 0 else 'empty'}, "
    #                 f"max={selection_value.max() if selection_value.numel() > 0 else 'empty'}")
    #             print(f"Next state value stats: min={next_state_value.min() if next_state_value.numel() > 0 else 'empty'}, "
    #                 f"max={next_state_value.max() if next_state_value.numel() > 0 else 'empty'}")
            
    #         assert all((~torch.isinf(scores.view(-1))) & (~torch.isnan(scores.view(-1)))), "Found inf or nan in scores"
            
    #         return {
    #             "rewards": scores,
    #             "inf_mask": inf_mask
    #         }