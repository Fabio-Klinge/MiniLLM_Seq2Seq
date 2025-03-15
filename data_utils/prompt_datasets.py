import random
import torch
import os
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size
from utils import print_rank
from tqdm import tqdm
import json


class PromptDataset(Dataset):
    '''
    Preprocesssing pipeline for the Conditional Generation dataset. (prompt, ground-truth response(D))
    Creates tensors from individual samples and moves them to GPU. 
    Distributed training setup. 
    Custom dataset class DistributedMMapIndexedDataset to handle large-scale data efficiently. 
    Memory-mapped files for memory-efficient and random access to data.
    '''
    def __init__(self, args, tokenizer, split, data_path=None, num=-1):
        super().__init__()
        self.tokenizer = tokenizer

        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length

        # Felxibility for different data types
        if args.bin_data:
            # Memory mapped binary data
            self.data = DistributedMMapIndexedDataset(data_path, f"{split}", get_rank(), get_world_size())
        elif args.json_data:
            # Json data
            self.data, self.origin_data = self.load_data_json(data_path)
        else:
            # Txt data
            self.data = self.load_data_txt(data_path)
        
        # Read conditional data from path
        # ? for different uses besides direct input to the model,
        # such as validation, examination, or other processing outside the standard training pipeline.
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            with open(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        elif os.path.exists(os.path.join(data_path, f"{split}.jsonl")):
            with open(os.path.join(data_path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        else:
            print_rank("WARNING: No answers exist")
        
        # Map from tokenized data to text representation
        self.label_map = {tokenizer.encode(x[0], add_special_tokens=False)[0]: x[0] for x in self.answers}
        
        # Instance count
        self.num = min(num, len(self.data)) if num > 0 else len(self.data)
        print_rank(f"Num instances: {len(self.data)}")
                    
    def __len__(self):
        return self.num

    def load_data_json(self, data_path):
        '''
        Formats different kinds of json data to create sample+label consistent for model.
        '''
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            data_path = os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")
        else:
            data_path = os.path.join(data_path, f"{self.split}.jsonl")
        
        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        print_rank("Loading Data")
        for d in tqdm(data_origin, disable=(get_rank() != 0)):
            prompt = d["prompt"].replace("<n>", "\n")
            prompt_ids = self.tokenizer.encode(prompt)
            output_ids = None
            if "output" in d:
                if isinstance(d["output"], list):
                    output_ids = self.tokenizer.encode(d["output"][0])
                else:
                    output_ids = self.tokenizer.encode(d["output"])
            # Model in/output
            data.append({
                "prompt_ids": prompt_ids,
                "output_ids": output_ids[:self.max_length - self.max_prompt_length]
            })
        print_rank("Load End")
        return data, data_origin

    def load_data_txt(self, data_path):
        with open(os.path.join(data_path, f"{self.split}.txt")) as f:
            lines = f.readlines()
        data = []
        print_rank("Loading Data")
        for line in lines:
            line = line.strip()
            line = line.replace("<n>", "\n")
            prompt = self.tokenizer.encode(line)
            data.append(prompt)
        print_rank("Load End")
        return data

    # Tokens to text
    def verbalizer(self):
        return self.label_map

    def __getitem__(self, index: int):
        '''
        Data sample from indexed dataset.
        '''
        data = self.data[index]
        if self.args.bin_data:
            data = data.astype(int)
        elif self.args.json_data:
            output_ids = data["output_ids"]
            data = data["prompt_ids"]
        
        prompt_length = self.max_prompt_length

        prompt = data[:prompt_length]
        rest = data[prompt_length:]  
        if self.args.json_data:
            if output_ids is not None:
                rest = output_ids  
    
        return index, prompt, rest
    
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

    def move_to_device(self, model_batch, no_model_batch, device):
        '''
        Move Data to GPU
        '''
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)        
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch
