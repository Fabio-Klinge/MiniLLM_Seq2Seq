import random
import torch
import os
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size
from utils import print_rank
from tqdm import tqdm
import json
import pickle
import numpy as np

class ANLIDataset(Dataset):
    '''
    Preprocesssing pipeline for the ANLI dataset. (premise, hypothesis, label)
    Creates tensors from individual samples and moves them to GPU. 
    Distributed training setup. 
    Custom dataset class DistributedMMapIndexedDataset to handle large-scale data efficiently. 
    Memory-mapped files for memory-efficient and random access to data.
    '''
    def __init__(self, args, tokenizer, split, data_path=None, num=-1):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.ratio = args.ratio
        self.pad_id = self.tokenizer.pad_token_id
        self.max_length = args.max_length
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length

        # Flexibility for different data types
        if args.bin_data:
            # Memory mapped binary data
            self.data = DistributedMMapIndexedDataset(data_path, f"{split}", get_rank(), get_world_size())
        elif args.json_data:
            # Json data
            self.data, self.origin_data = self.load_data_json(data_path)
        else: 
            print_rank("WARNING: No data exists")
        
        # Read data for label map
        if os.path.exists(os.path.join(data_path, f"{self.split}.jsonl")):
            with open(os.path.join(data_path, f"{self.split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x[""]] for x in self.raw]
        elif os.path.exists(os.path.join(data_path, f"{split}.jsonl")):
            with open(os.path.join(data_path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["response"] if isinstance(x["response"], list) else [x["response"]] for x in self.raw]
        else:
            print_rank("WARNING: No answers exist")
        ### only recognizes sequences by first token ###
        # Map from tokenized data to text representation
        self.label_map = {tokenizer.encode(x[:2], add_special_tokens=False)[0]: x[0] for x in self.answers}

        # Instance count
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print_rank(f"Num instances: {len(self.data)}")

       
    # Tokens to text
    def verbalizer(self):
        return self.label_map

    def __len__(self):
        return self.num

    def load_data_json(self, data_path):
        '''
        Formats different kinds of json data to create sample+label consistent for model.
        '''
        if os.path.exists(os.path.join(data_path, f"{self.split}.jsonl")):
            data_path = os.path.join(data_path, f"{self.split}.jsonl")
        else:
            print_rank("WARNING: Data path for json does not exist")

        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        print_rank("Loading Data")
        for d in tqdm(data_origin, disable=(get_rank() != 0)):
            prompt = d["prompt"]
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = None
            if "response" in d:
                if isinstance(d["response"], list):
                    response_ids = self.tokenizer.encode(d["response"][0])
                else:
                    response_ids = self.tokenizer.encode(d["response"])
            # Model in/response
            data.append({
                "prompt_ids": prompt_ids,
                "response_ids": response_ids[:self.max_length - self.max_prompt_length]
            })
        print_rank("Load End")
        return data, data_origin

    def __getitem__(self, index: int):
        '''
        Data sample from indexed/json dataset.
        '''
        data = self.data[index]
    
        if self.args.bin_data:
            data = data.astype(int)
        elif self.args.json_data:
            response_ids = data["response_ids"]
            data = data["prompt_ids"]
        
        prompt_length = self.max_prompt_length

        # Assumes prompt to be exactly max_prompt_length
        prompt = data[:prompt_length]
        #padding? # Prompts have to be truncated to max prompt length before, padded if too small
        # What represents self.pad_id
        rest = data[prompt_length:]  
        if self.args.json_data:
            if response_ids is not None:
                rest = response_ids  
    
        return index, prompt, rest
    
    # Represents labels in binary data
    def collate(self, samples):
        '''
        Organizes & output individual data batches. 
        Handles padding, calls _process_lm for each sample, 
        '''
        # Batch size ;)
        bs = len(samples)
        max_prompt_length = self.max_prompt_length
        max_rest_length = max([len(samp[2]) for samp in samples])

        # Tensor inits and padding
        model_batch = {
            "input_ids": torch.ones(bs, max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_prompt_length, dtype=torch.long),
            # "position_ids": torch.zeros(bs, max_prompt_length, dtype=torch.long)
        }
        no_model_batch = {
            # Indices of samples
            "idx": torch.zeros(bs, dtype=torch.long),
            # Init for rest of the input data
            "rest_ids": torch.ones(bs, max_rest_length, dtype=torch.long) * self.pad_id
        }
        
        # Fill tensors with data
        for i, (idx, prompt, rest) in enumerate(samples):
            # left padding
            model_batch["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
            # Update attention mask
            model_batch["attention_mask"][i][-len(prompt):] = 1
            # model_batch["position_ids"][i][-len(prompt):] = torch.arange(len(prompt))
            no_model_batch["idx"][i] = idx
            # Save the rest of the i'th sample to a tensor
            no_model_batch["rest_ids"][i][:len(rest)] = torch.tensor(rest, dtype=torch.long)
        
        # Ready for model
        return model_batch, no_model_batch
            



    