import os
import json
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from transformers import mpu
import torch.distributed as dist

from data_utils.distributed_indexed import DistributedMMapIndexedDataset
from data_utils.indexed_dataset import best_fitting_dtype
from torch.distributed import get_rank, get_world_size
from utils import print_rank

# pipeline so das model data nur input und attention mask hat, no model data hat labels und loss mask
# Label für forward pass ist model output (eventuell dummy label für erste iteration, maybe echt label dann noch model_data)
class PPOPipeline():
    def __init__(self, args, tokenizer, split, ppo_data_path=None, fix_prompts=False, num=-1):
        super().__init__()
        self.tokenizer = tokenizer
    
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.pad_token_id
        self.max_length = args.max_length
        self.rng_ppo = random.Random(args.seed_ppo)
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length

        self.ppo_ctx = DistributedMMapIndexedDataset(ppo_data_path, f"{split}", get_rank(), get_world_size())
        self.ppo_raw, self.ppo_answers = None, None
        if os.path.exists(os.path.join(ppo_data_path, f"{split}.jsonl")):
            with open(os.path.join(ppo_data_path, f"{split}.jsonl")) as f:
                self.ppo_raw = [json.loads(line) for line in f.readlines()]
                self.ppo_answers = [x["label"] if isinstance(x["label"], list) else [x["label"]] for x in self.ppo_raw]

        self.num = min(num, len(self.ppo_ctx)) if num > 0 else len(self.ppo_ctx)
        self.fix_prompts = fix_prompts
        self.prompt_lengths = [None for _ in range(num)]
        print_rank(f"Num PPO instances: {len(self.ppo_ctx)}")
            
    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        #before called data!!!
        sample = self.ppo_ctx[index].astype(int)

        # print("### Sample in getitem", sample)
        # #print("### Sample in getitem decoded", self.tokenizer.decode(sample, skip_special_tokens=False))
        # vocab_size = self.tokenizer.vocab_size
        # print(f"Tokenizer vocabulary size: {vocab_size}")
        
        # # Find any out of range tokens
        # invalid_tokens = [(i, t) for i, t in enumerate(sample) if t >= vocab_size]
        # if invalid_tokens:
        #     print(f"Found invalid tokens (position, value): {invalid_tokens}")
        
        # # Filter out invalid tokens for decoding
        # valid_sample = [t for t in sample if t < vocab_size]
        # print(self.tokenizer.decode(valid_sample, skip_special_tokens=False))
            
        #assert len(sample) <= self.max_prompt_length
        
        # Represents max integer in tokenizer vocab (2^18-1)
        if 65535 in sample:
            # Dest. of seperator index
            source_len = np.where(sample==65535)[0][0]
            label_ids = sample[source_len+1:]
            input_ids = sample[:source_len]
            #prompt = input_ids[:source_len]
            # Sequence reconstruted without special token
            #input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
            #print("meparanoid", self.tokenizer.decode(input_ids, skip_special_tokens=False),"actual_label", self.tokenizer.decode(label_ids, skip_special_tokens=False))
            self.args.temp_sample = {"input_ids": input_ids, "label": label_ids}

        else:
            print("### No special token found in input_ids in ppopipeline, 65535 not consistently in data")
            #input_ids = sample["input_ids"]
            #label_ids = sample["label"]
            return self.args.temp_sample["input_ids"], self.args.temp_sample["label"]

        #print(f"Length of label_ids: {len(label_ids)}")

        # return prompt, rest
        return input_ids, label_ids
    
    def collate(self, samples):
        '''
        Organizes & outputs individual data batches. 
        Handles padding, calls _process_lm for each sample, 
        '''
        bs = len(samples)
        
        # Tensor inits for right padded data
        model_batch = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length),
            #"labels": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
        }
        
        no_model_batch = {
            # -100 ignored by torch loss functions
            "labels": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * -100,
            #"loss_mask": torch.zeros(bs, self.max_prompt_length)
        }
        
        for i, (input_ids, label) in enumerate(samples):
            # Truncate input_ids before assignment if needed
            if len(input_ids) > self.max_prompt_length:
                input_ids = input_ids[:self.max_prompt_length]
            
            # Now assign the (potentially truncated) input_ids
            model_batch["input_ids"][i][:len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
            model_batch["attention_mask"][i][:len(input_ids)] = 1.0
            
            # Handle labels if present
            if label is not None:
                if len(label) > self.max_prompt_length:
                    label = label[:self.max_prompt_length]
                no_model_batch["labels"][i][:len(label)] = torch.tensor(label, dtype=torch.long)
                #no_model_batch["loss_mask"][i][:len(label)] = 1.0


        # for i, (input_ids , label) in enumerate(samples):
        #     # Truncate input_ids if too long
        #     if model_batch["input_ids"][i].size(0) > self.max_prompt_length:
        #         print("### Truncating input_ids", model_batch["input_ids"][i].size(0))
        #         print("without size", model_batch["input_ids"][i])
        #         model_batch["input_ids"][i] = model_batch["input_ids"][i][:self.max_prompt_length]
        #         model_batch["attention_mask"][i] = model_batch["attention_mask"][i][:self.max_prompt_length]

        #     # Right-pad input_ids
        #     model_batch["input_ids"][i][:len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
        #     model_batch["attention_mask"][i][:len(input_ids)] = 1.0
        #     #model_batch["labels"][i][:len(label)] = torch.tensor(label, dtype=torch.long)

        #     # Truncate input_ids if too long
        #     if model_batch["input_ids"][i].size(0) > self.max_prompt_length:
        #         print("### Truncating input_ids", model_batch["input_ids"][i].size(0))
        #         print("without size", model_batch["input_ids"][i])
        #         model_batch["input_ids"][i] = model_batch["input_ids"][i][:self.max_prompt_length]
        #         model_batch["attention_mask"][i] = model_batch["attention_mask"][i][:self.max_prompt_length]

        #     # Delete and check where its used (not really needed)
        #     if label is not None:
        #         no_model_batch["labels"][i][:len(label)] = torch.tensor(label, dtype=torch.long)
        #         no_model_batch["loss_mask"][i][:len(label)] = 1.0
        #         # Truncate label if too long
        #         if no_model_batch["labels"][i].size(0) > self.max_prompt_length:
        #             no_model_batch["labels"][i] = no_model_batch["labels"][i][:self.max_prompt_length]
        #             no_model_batch["loss_mask"][i] = no_model_batch["loss_mask"][i][:self.max_prompt_length]
        
        return model_batch, no_model_batch

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        if self.args.model_parallel:
            dp_world_size = mpu.get_data_parallel_world_size()
            dp_rank = mpu.get_data_parallel_rank()
        else:
            dp_world_size = dist.get_world_size()
            dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )


class LMPipeline():
    def __init__(self, args, tokenizer, split, lm_data_path=None, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        # Represents max integer in tokenizer vocab (2^18-1)    
        self.split_id = np.iinfo(best_fitting_dtype(len(tokenizer))).max
        self.pad_id = self.tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.rng_lm = random.Random(args.seed_lm)

        self.lm_ctx = DistributedMMapIndexedDataset(lm_data_path, f"{split}", get_rank(), get_world_size())
        self.num = min(num, len(self.lm_ctx)) if num > 0 else len(self.lm_ctx)
        print_rank(f"Num LM instances: {len(self.lm_ctx)}")
            
    def __len__(self):
        return self.num

    def __getitem__(self, index):
        return self._get_lm(index)

    def _get_lm(self, index):
        # Data retrieval for distributed training
        data = self.lm_ctx[index]
        #print("### Data in getitem", data)
        
        # Convert to integer array
        input_ids = data.astype(int)
        
        return {
            "input_ids": input_ids[:self.max_prompt_length]
        }
        
        # Find the index of -1
        separator_index = np.where(int_data == -1)[0]
        

        # Always true so far
        if len(separator_index) == 0:
            #print("### No separator found in data")
            # If there's no separator, treat all as input_ids
            return {
                "sample": int_data,
            }
        
        # The first occurrence of -1 is our separator
        separator_index = separator_index[0]
        
        # Split the data
        input_ids = int_data[:separator_index]
        label = int_data[separator_index + 1:]  # +1 to skip the -1 separator
        
        return {
            "input_ids": input_ids,
            "label": label
        }
             #print("### prompt value in process_lm. If true remove prompt handle###", prompt)
        
        # if 65535 in sample:
        #     print("###special token in eval data as well###")
        #     # Dest. of seperator index
        #     source_len = np.where(sample==65535)[0][0]
        #     label_ids = sample[source_len+1:]
        #     input_ids = sample[:source_len]
        #     #prompt = input_ids[:source_len]
        #     # Sequence reconstruted without special token
        #     #input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        #     #print("meparanoid", self.tokenizer.decode(input_ids, skip_special_tokens=False),"actual_label", self.tokenizer.decode(label_ids, skip_special_tokens=False))
        #     self.args.temp_sample = {"input_ids": input_ids, "label": label_ids}

        # else:
        #     print("### No special token found in input_ids, 65535 not consistently in data")
        #     #input_ids = sample["input_ids"]
        #     #label_ids = sample["label"]
        #     input_ids = self.args.temp_sample["input_ids"]
        #     label_ids = self.args.temp_sample["label"]
        
        # input_ids = input_ids[:self.max_prompt_length]
        # input_len = len(input_ids)   
    def _process_lm(self, 
                    i: int, 
                    samp: dict[str, np.ndarray],
                    model_data: dict[str, torch.Tensor], 
                    no_model_data: dict[str, torch.Tensor]):
        '''
        Preprocesses individual samples and organizes them into structured data (tensors). 
        Handles special tokens, dynamic input length & data extraction.
        '''

    
        input_ids = samp["input_ids"]
        #print("### Input ids in process_lm", input_ids)
        
        # Handle split_id if present
        if self.split_id in input_ids:
            source_len = np.where(input_ids==self.split_id)[0][0]
            input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        
        # Truncate to max length
        input_ids = input_ids[:self.max_prompt_length]
        input_len = len(input_ids)
        
        # Calculate split point (middle of sequence)
        split_point = input_len // 2
        
        # Fill input tensors with first half
        model_data["input_ids"][i][:split_point] = torch.tensor(input_ids[:split_point], dtype=torch.long)
        model_data["attention_mask"][i][:split_point] = 1.0
        #print("### Model data in process_lm", model_data["input_ids"][i])
        
        # Fill label tensors with second half
        no_model_data["labels"][i][:input_len-split_point] = torch.tensor(input_ids[split_point:], dtype=torch.long)
        no_model_data["labels"][i][input_len-split_point:] = -100  # padding ignored by loss
        #print("### No model data in process_lm", no_model_data["labels"][i])
        # #print("### Sample in process_lm", samp)
        # input_ids = samp["input_ids"]
        # source_len = 1  
        # prompt = None


        # # label_len = len(label_ids)
        # if self.split_id in input_ids:
        #     source_len = np.where(input_ids==self.split_id)[0][0]
        #     input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)
        # input_ids = input_ids[:self.max_prompt_length]
        # input_len = len(input_ids)
        # print("input_len", input_len)
        
        # model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        # model_data["attention_mask"][i][:input_len-1] = 0.0
        # model_data["attention_mask"][i][input_len:] = self.pad_id
        
        # no_model_data["labels"][i][:input_len] = torch.tensor(input_ids, dtype=torch.long)
        # #print("attention mask", model_data["attention_mask"][i])
        # #print ("### Model data in process_lm", model_data.shape, "\n length of attention mask", model_data["attention_mask"][i][:input_len-1].shape)
        

        # # Add debug prints to verify shapes
        # print(f"Shape of input_ids: {model_data['input_ids'][i].shape}")
        # print(f"Shape of attention_mask: {model_data['attention_mask'][i].shape}")
        #print(f"input_len_adjusted: {input_len_adjusted}")
            
        # # Right-pad input_ids
        # model_data["input_ids"][i][:input_len] = torch.tensor(input_ids, dtype=torch.long)
        # model_data["attention_mask"][i][:input_len] = 1.0
        # #model_data["labels"][i][:label_len] = torch.tensor(label_ids, dtype=torch.long)
        
        # # Positional IDs for GPT2
        # if self.args.model_type in ["gpt2"]:    
        #     model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)

        # #has to be as long as label tensor, is already padded
        # # Right-pad labels/label
        # no_model_data["labels"][i][:label_len] = torch.tensor(label_ids, dtype=torch.long)
        # no_model_data["loss_mask"][i][:label_len] = 1.0
        
        # # Set prompt tokens to -100 in labels
        # if prompt is not None:
        #     no_model_data["labels"][i][:source_len-1] = -100
        #     #no_model_data["loss_mask"][i][:source_len-1] = 0

        # # Handle prompt for generation
        # if prompt is not None:
        #     gen_data["input_ids"][i][:len(prompt)] = torch.tensor(prompt, dtype=torch.long)
        #     gen_data["attention_mask"][i][:len(prompt)] = 1.0
            
    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch


    def collate(self, samples):
        '''
        Organizes & outputs individual data batches. 
        Handles padding, calls _process_lm for each sample, _ma
        '''
        bs = len(samples)
        
        # Tensor inits for right padded data
        model_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long)
            #"labels": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id
        }
        
        no_model_data = {
            # -100 ignored by torch loss functions
            "labels": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * -100,
            #"loss_mask": torch.zeros(bs, self.max_prompt_length)
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }
        
        # Fill Tensors (special token handling for padded data)
        for i, samp in enumerate(samples):
            self._process_lm(i, samp, model_data, no_model_data)
            
        if self.args.small_dataset == 0 or self.args.small_dataset == 1 or self.args.small_dataset == 2:
            print("### Model data in collate", model_data)
            print("### No model data in collate", no_model_data)
            self.args.small_dataset += 1

        # print("### Model data in collate", model_data)
        # print("### No model data in collate", no_model_data)
        
        # #Returns processed data
        # print("### Model data in collate", model_data, no_model_data)
        return model_data, no_model_data

    def create_loader(self, batch_size: int, shuffle=False, drop_last: bool = False, num_workers: int = 0) -> DataLoader:
        if self.args.model_parallel:
            dp_world_size = mpu.get_data_parallel_world_size()
            dp_rank = mpu.get_data_parallel_rank()
        else:
            dp_world_size = dist.get_world_size()
            dp_rank = dist.get_rank()
        
        sampler = DistributedSampler(self, shuffle=shuffle, drop_last=drop_last, rank=dp_rank, num_replicas=dp_world_size)
        return DataLoader(
            
            self, sampler=sampler, batch_size=batch_size, collate_fn=self.collate, num_workers=num_workers
        )
