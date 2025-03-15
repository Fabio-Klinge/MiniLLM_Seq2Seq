import random
import torch
import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size, barrier
from utils import print_rank
from utils import save_rank


class LMTrainDataset(Dataset):
    '''
    Preprocesssing pipeline for the pre-training corpus (long-document plain text (Dpt))
    Creates tensors from individual samples and moves them to GPU. 
    Distributed training setup. 
    Custom dataset class DistributedMMapIndexedDataset to handle large-scale data efficiently. 
    Memory-mapped files for memory-efficient and random access to data.
    '''
    def __init__(self, args, tokenizer, path, split, num, ratio, rng_sample: random.Random):
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.pad_id = self.tokenizer.pad_token_id
        # For gpt2
        #self.pad_id = self.tokenizer.eos_token_id
        self.ratio = ratio
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.rng_sample = rng_sample
        print_rank(f"Loading data from {path}")
        self.lm_ctx = DistributedMMapIndexedDataset(path, f"{split}", get_rank(), get_world_size())
        
        # Save data split
        if os.path.exists(os.path.join(path, f"{split}.jsonl")):
            with open(os.path.join(path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["label"] if isinstance(x["label"], list) else [x["label"]] for x in self.raw]
        
        print_rank(len(self.lm_ctx))
        if num == -1:
            self.num = len(self.lm_ctx)
        else:
            self.num = num

        print_rank(f"Num LM instances: {len(self.lm_ctx)}")

    def __len__(self):
        return self.num
    
    # Implicitly called for indexing operations on LMTrainDataset
    def __getitem__(self, index):
        return self._get_lm(index)
    
    # # Get model input data from indexed dataset
    # def _get_lm(self, index):
    #     # Data retrieval for distributed training
    #     data = self.lm_ctx[index]
    #     print("### Data in getitem", data)
    #     # Tokenized data
    #     input_ids = data.astype(int)
    #     #print("### Data after conversion", input_ids)
        
    #     return {
    #         "input_ids": input_ids
    #     }
    

    def _get_lm(self, index):
        # Data retrieval for distributed training
        data = self.lm_ctx[index]
        #print("### Data in getitem", data)
        
        # Convert to integer array
        int_data = data.astype(int)

        return {
            "sample": int_data,
        }


    def _process_lm(self, i, samp, model_data, no_model_data, gen_data):
        '''
        Preprocesses individual samples and organizes them into structured data (tensors). 
        Handles special tokens, dynamic input length & data extraction.
        Requires (model_data, no_model_data, gen_data) to be initialized and synchronized in size and purpose.
        '''
        #print("### Sample in process_lm", samp)
        sample = samp["sample"]
        source_len = 1  
        prompt = None

        #print("### prompt value in process_lm. If true remove prompt handle###", prompt)
        #print("### Sample in collate", sample, type(sample))
        if 65535 in sample:
            #print("### Special token found in input_ids")
            # Dest. of special token
            separator_indices = np.where(sample==65535)[0]
            #print("### Separator indeces in process_lm", separator_indices)
            label_ids = sample[source_len+1:]
            input_ids = sample[:source_len]
            # Split the data
            # input_ids = sample[:separator_indeces[0]]
            # label_ids = sample[separator_indeces[0] + 1:]  # +1 to skip the -1 separator
            # uid = sample[separator_indeces[1] + 1:]
            # premise = sample[separator_indeces[2] + 1:]
            # hypothesis = sample[separator_indeces[3] + 1:]

            input_ids = sample[:separator_indices[0]]
            label_ids = sample[separator_indices[0] + 1:separator_indices[1]]
            uid = sample[separator_indices[1] + 1:separator_indices[2]]
            premise = sample[separator_indices[2] + 1:separator_indices[3]]
            hypothesis = sample[separator_indices[3] + 1:separator_indices[4]]
            classification_label = sample[separator_indices[4] + 1:]
            #prompt = input_ids[:source_len]
            # Sequence reconstruted without special token
            #input_ids = np.concatenate([input_ids[:source_len], input_ids[source_len+1:]], axis=0)

        else:
            print("### No special token found in sample, bin data corrupted ###")
        
        input_ids = input_ids[:self.max_length]
        label_len = label_ids[:self.max_length]
        input_len = len(input_ids)
        label_len = len(label_ids)
        if input_len >= self.max_length:
            print(f"anli inp can be larger than {self.max_length}:", input_len)
        if label_len >= self.max_length:
            print(f"anli lbl can be larger than {self.max_length}:", label_len)
        
        
        # Right-pad input_ids
        model_data["input_ids"][i][:input_len] = torch.tensor(input_ids, dtype=torch.long)
        model_data["attention_mask"][i][:input_len] = 1.0
        
        # Positional IDs for GPT2
        if self.args.model_type in ["gpt2"]:    
            model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)

        #has to be as long as label tensor, is already padded
        # Right-pad labels/label
        no_model_data["labels"][i][:label_len] = torch.tensor(label_ids, dtype=torch.long)
        no_model_data["uids"][i][:len(uid)] = torch.tensor(uid, dtype=torch.long)
        no_model_data["premises"][i][:len(premise)] = torch.tensor(premise, dtype=torch.long)
        no_model_data["hypotheses"][i][:len(hypothesis)] = torch.tensor(hypothesis, dtype=torch.long)
        #no_model_data["classification_labels"][i][:len(classification_label)] = torch.tensor(classification_label, dtype=torch.long)



        # Explicitly set the first token to self.pad_id
        #no_model_data["labels"][i][0] = self.pad_id
        #no_model_data["loss_mask"][i][:label_len] = 1.0
        
        # Set prompt tokens to -100 in labels
        if prompt is not None:
            no_model_data["labels"][i][:source_len-1] = -100
            #no_model_data["loss_mask"][i][:source_len-1] = 0

        # Handle prompt for generation
        if prompt is not None:
            gen_data["input_ids"][i][:len(prompt)] = torch.tensor(prompt, dtype=torch.long)
            gen_data["attention_mask"][i][:len(prompt)] = 1.0

        # torch.set_printoptions(profile="full")
        # print("### Model data in process_lm", model_data)
        # print("### No model data in process_lm", no_model_data)
        # torch.set_printoptions(profile="default")
            
            
            
        # # Assignment to tensors
        # model_data["input_ids"][i][:input_len-1] = torch.tensor(input_ids[:-1], dtype=torch.long)
        # # Attention mask for padded tokens
        # model_data["attention_mask"][i][:input_len-1] = 1.0

        # # Positional IDs for GPT2
        # if self.args.model_type in ["gpt2"]:    
        #     model_data["position_ids"][i][:input_len-1] = torch.arange(0, input_len-1, dtype=torch.long)
        
        # # Label tensor, offset by one position for prediction
        # no_model_data["label"][i][:input_len-1] = torch.tensor(input_ids[1:], dtype=torch.long)
        # # Initial prompt moved
        # no_model_data["label"][i][:source_len-1] = -100
        # # Created to ignore specific tokens of model output
        # no_model_data["loss_mask"][i][:input_len-1] = 1.0
        # no_model_data["loss_mask"][i][:source_len-1] = 0
        
        # # Prompt saved seperately
        # if prompt is not None:
        #     gen_data["input_ids"][i][-len(prompt):] = torch.tensor(prompt, dtype=torch.long)
        #     gen_data["attention_mask"][i][-len(prompt):] = 1.0

    def move_to_device(self, model_data, no_model_data, gen_data, device):
        '''
        Move Data to GPU
        '''
        # Input
        for k in model_data:
            model_data[k] = model_data[k].to(device)

        # Labels
        for k in no_model_data:
            no_model_data[k] = no_model_data[k].to(device)

        # Prompt
        for k in gen_data:
            gen_data[k] = gen_data[k].to(device)

        # Updated dicts with data- and mask tensors
        # Returned so tensors already placed on GPU do not have to be moved
        #check print rank
        return model_data, no_model_data, gen_data

    def collate(self, samples):
        '''
        Organizes & outputs individual data batches. 
        Handles padding, calls _process_lm for each sample, 
        '''
        bs = len(samples)
        max_length = self.max_length
        
        # Tensor inits for right padded data
        model_data = {
            "input_ids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, max_length),
        }

        no_model_data = {
            # -100 ignored by torch loss functions
            "labels": torch.ones(bs, max_length, dtype=torch.long) * -100,
            "uids": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "premises": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            "hypotheses": torch.ones(bs, max_length, dtype=torch.long) * self.pad_id,
            #"classification_labels": torch.ones(bs, dtype=torch.long) * self.pad_id,
        }
        
        gen_data = {
            "input_ids": torch.ones(bs, self.max_prompt_length, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_prompt_length, dtype=torch.long),
        }
        
        # Tensor creation & special token handling for padded data
        for i, samp in enumerate(samples):
            #print("### Sample in collate", samp, type(samp))
            self._process_lm(i, samp, model_data, no_model_data, gen_data)

        # print("### Model data in collate", model_data)
        # print("### No model data in collate", no_model_data)
        
        # Returns processed data
        return model_data, no_model_data, gen_data
