import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from fairseq.optim.adafactor import Adafactor
import deepspeed

import matplotlib.pyplot as plt
import numpy as np

import random
import json
from tqdm import tqdm
import math

from collections import Counter

from transformers import (
    AutoModelForSeq2SeqLM,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    mpu,
    GenerationConfig)


from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingLR

from arguments import get_args

from data_utils.lm_datasets import LMTrainDataset
from utils import get_optimizer_params, get_optimizer_params_peft, print_args, initialize
from utils import print_rank, get_rank
from utils import save_rank
from utils import all_gather
from utils import load_parallel, save_parallel
from utils import get_tokenizer, get_model, parallel_model_map

from accelerate import init_empty_weights

from rouge_metric import compute_metrics

# Parameter efficient fine-tuning
from peft import PeftModel

torch.set_num_threads(4)




def get_optimizer(args, model):
    """Set up the optimizer."""

    # DistributedDataParallel
    while isinstance(model, DDP):
        model = model.module

    # Build parameter groups (weight decay and non-decay).
    if args.peft is not None:
        param_groups = get_optimizer_params_peft(args, model)
    else:
        param_groups = get_optimizer_params(args, model)

    # Use AdamW.
    #optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = Adafactor(param_groups, lr=args.lr, weight_decay=args.weight_decay, relative_step=False, scale_parameter=False, warmup_init=False)

    print_rank(f'Optimizer = {optimizer.__class__.__name__}')
    return optimizer


def get_learning_rate_scheduler(args, optimizer):
    '''
    Offers constant, cosine and noam lr schedules with warmup.
    '''
    if args.total_iters is None:
        args.total_iters = args.train_iters_per_epoch * args.epochs
    if args.lr_decay_style == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters)
    elif args.lr_decay_style == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.total_iters,
            eta_min=args.lr_min)
    elif args.lr_decay_style == "noam":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_iters,
            num_training_steps=args.total_iters,
            power=0.5)
    else:
        raise ValueError(f"lr_scheduler of type {args.lr_decay_style} is not supported yet.")

    return lr_scheduler


def setup_model_and_optimizer(args, ds_config, device, set_optim=True):
    # get the model
    model = get_model(args, device)
    # get the optimizer and lr_scheduler
    if set_optim:
        optimizer = get_optimizer(args, model)
        #lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    else:
        optimizer = None
        
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        mpu=mpu if args.model_parallel else None,
        config_params=ds_config
    )
    
    # get the memory usage
    print_rank("Model mem\n", torch.cuda.memory_summary())

    
    return model, optimizer


def prepare_dataset(args, tokenizer):
    data = {}
    # Random number generator object
    # Precisely this is PRNG: Same seed will always lead to the same sequence of numbers (deterministic)
    rng_sample = random.Random(args.seed)
    if args.do_train:
        # Constructor of LMTrainDataset loads data and saves dict to retrieve data sample-wise
        data["train"] = LMTrainDataset(args, tokenizer, os.path.join(args.data_dir, args.anli_round, "train/"), "train", args.train_num, args.train_ratio, rng_sample)
        print_rank("train num", len(data["train"]))
        data["dev"] = LMTrainDataset(args, tokenizer, os.path.join(args.data_dir, args.anli_round, "train/"), "dev", args.dev_num, args.dev_ratio, rng_sample)
    elif args.do_eval:
        data["test"] = LMTrainDataset(args, tokenizer, os.path.join(args.data_dir, args.anli_round, "dev/"), "dev", args.dev_num, args.dev_ratio, rng_sample)
    else:
        raise ValueError("Do train and do eval must set one")
    return data


def get_distil_loss(args, tokenizer, model, teacher_model, model_batch, no_model_batch, logits):
    '''
    Word-level KD.
    Computes soft cross entropy loss between student and teacher logits.
    '''
    with torch.no_grad():
        teacher_model.eval()
        teacher_outputs = teacher_model(**model_batch, use_cache=False)
        teacher_logits = teacher_outputs.logits
    if args.model_parallel:
        distil_losses = mpu.parallel_soft_cross_entropy_loss(logits.float(), teacher_logits.float())
        distil_losses = distil_losses.view(-1)
        loss_mask = no_model_batch["loss_mask"].view(-1)
        distil_loss = (distil_losses * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (no_model_batch["labels"] != -100).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    
    return distil_loss


def get_teacher_lm_loss(args, tokenizer, model, teacher_model, model_batch):
    '''
    Teacher loss on input batch for KD
    '''
    with torch.no_grad():
        t_gen_out = teacher_model.generate(
            **model_batch,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.max_length,
            top_k=0,
            top_p=1,
            temperature=1.0,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=False)
    
    full_ids = t_gen_out.sequences

    # Split to input & labels for student
    input_ids = full_ids[:, :-1]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels = full_ids[:, 1:]    
    labels = torch.masked_fill(labels, mask==0, -100)
    labels[:, :model_batch["input_ids"].size(1)-1] = -100
    #loss_mask = (labels != -100).float()
    
    new_batch = {
        "input_ids": input_ids,
        "attention_mask": mask,
    }
    
    # Positional encoding
    if args.model_type in ["gpt2"]:
        position_ids = torch.cumsum(mask, dim=-1) - 1
        position_ids = torch.masked_fill(position_ids, mask==0, 0)    
        new_batch["position_ids"] = position_ids    
    
    # -100 is ignored in loss computation
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    outputs = model(**new_batch, return_dict=True, use_cache=False)
    logits = outputs.logits
    lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

    return lm_loss

def get_plot(args, metrics):
    """
    Creates a visualization of training metrics for T5 finetuning.
    
    Args:
        metrics (list): List of dictionaries containing training metrics
        
    Returns:
        matplotlib.figure.Figure: Figure containing the plots
    """
    # Move tensors to cpu for plotting (for nested dict)
    metrics = [{k: v.cpu() if torch.is_tensor(v) else v for k, v in d.items()} for d in metrics]

    # Convert list of dictionaries to dictionary of lists
    plot_data = {key: [d[key] for d in metrics] for key in metrics[0].keys()}
    
    # Create figure with subplots
    plt.clf()  # Clear any existing plots
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout with more space
    gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3, top=0.95)
    
    fig.suptitle('T5 Finetuning Progress on ANLI', fontsize=14, y=0.98)
    
    # Plot 1: Loss curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(plot_data['step'], plot_data['loss'], label='Total Loss', color='blue')
    ax1.plot(plot_data['step'], plot_data['classification_loss'], 
             label='Classification Loss', color='red', linestyle='--')
    ax1.plot(plot_data['step'], plot_data['reason_loss'], 
            label='Reason Loss', color='green', linestyle='--')
    ax1.set_title('Training Losses')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning rate
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(plot_data['step'], plot_data['accuracy'], color='green')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Optimizer scale (if available)
    ax3 = fig.add_subplot(gs[1, 1])
    if any(plot_data['new_accuracy']):  # Only plot if scale is non-zero
        ax3.plot(plot_data['step'], plot_data['new_accuracy'], color='purple')
        ax3.set_title('Token Level Accuracy')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Accuracy')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Step time
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(plot_data['step'], plot_data['step_time'], color='orange')
    ax4.set_title('Step Time')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Time (s)')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Training progress
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(args.sequence_level_acc[0], args.sequence_level_acc[1], color='brown')
    ax5.set_title('Sequence Level Accuracy (Last Evaluation)')
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Accuracy')
    ax5.grid(True, alpha=0.3)

    plt.close(fig)
    return fig

def check_model_inp(tensor, tensor_name, max_threshold=1e5, min_treshold=-1e5):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {tensor_name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {tensor_name}")
    if torch.abs(tensor).max() > max_threshold or torch.abs(tensor).min() < min_treshold:
        max_value = torch.abs(tensor).max().item()
        print(f"Warning: High or low value ({max_value}) detected in {tensor_name}")
    if 65535 in tensor:
        print(f"Warning: 65535 detected in {tensor_name}")

    else:
        print(f"No NaN, Inf, or high values in {tensor_name}")
    
    # Additional statistics
    print(f"  Shape: {tensor.shape}")
    print(f"  Data type: {tensor.dtype}")
    print(f"  Min value: {tensor.min().item()}")
    print(f"  Max value: {tensor.max().item()}")
    print(f"  Mean value: {tensor.float().mean().item()}")
    print(f"  Standard deviation: {tensor.float().std().item()}")
    print(f"  Non-zero elements: {torch.count_nonzero(tensor).item()}")
    
def check_input_batch(batch, iteration):
    """Check input batch for NaN/Inf values"""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN detected in input {key} at iteration {iteration}")
                return True
            if torch.isinf(value).any():
                print(f"Inf detected in input {key} at iteration {iteration}")
                return True
    return False
    

def check_gradients(model_engine):
    """Check gradients for NaN/Inf values in DeepSpeed engine"""
    has_nan = False
    has_inf = False
    
    for name, param in model_engine.module.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"Inf gradient detected in {name}")
                has_inf = True
    
    return has_nan or has_inf   


def compute_class_reas_loss(args, logits, labels, device, seq_len, loss_func, compute_acc=False, tokenizer=None, it=None):
    """
    Compute classification and reasoning loss for ANLI.
    
    Args:
        logits (torch.Tensor): Model predictions
        labels (torch.Tensor): Target labels
        seq_len (int): Sequence length
        
    Returns:
        torch.Tensor: Classification loss
        torch.Tensor: Reasoning loss
    """
    discard_loss = 0

    # Split the logits and labels for the first token and the rest
    first_token_logits = logits[:, 0, :]
    rest_tokens_logits = logits[:, 1:, :]
    first_token_labels = labels[:, 0]
    rest_tokens_labels = labels[:, 1:]
    
    # Compute loss for the first token and the rest
    classification_token_loss = loss_func(first_token_logits, first_token_labels).mean()
    reason_token_loss = loss_func(rest_tokens_logits.reshape(-1, rest_tokens_logits.size(-1)), rest_tokens_labels.reshape(-1)).mean()

    # Combine the two losses, giving them equal weight
    loss = args.loss_ratio * classification_token_loss.mean() + (1 - args.loss_ratio) * reason_token_loss.mean()
    
    new_acc = (first_token_labels == torch.argmax(first_token_logits, dim=-1))
    # Compute accuracy for the first token
    if it % 1000 == 0:
        print("token level acc raw", new_acc)
        print("first_token_labels",tokenizer.decode(first_token_labels), first_token_labels.shape,"\n")
        print("class predictions",tokenizer.decode(torch.argmax(first_token_logits, dim=-1)), torch.argmax(first_token_logits, dim=-1), "\n")
        print("first prediciton", tokenizer.decode(torch.argmax(logits[0,:,:].reshape(-1, logits[0,:,:].size(-1)), dim=-1), skip_special_tokens=False),"\n")
        stoopid = labels[0,:]
        print("first label:", tokenizer.decode(stoopid[stoopid != -100], skip_special_tokens=False),"\n")
        print("rest_tokens_logits", tokenizer.decode(torch.argmax(rest_tokens_logits.reshape(-1, rest_tokens_logits.size(-1)), dim=-1), skip_special_tokens=False),"\n")
        valid_mask = (rest_tokens_labels.reshape(-1) >= 0) & (rest_tokens_labels.reshape(-1) < len(tokenizer))

        # Apply the mask to get only valid tokens
        valid_tokens = rest_tokens_labels.reshape(-1)[valid_mask]
        print("rest_tokens_labels", tokenizer.decode(valid_tokens, skip_special_tokens=False))


    new_acc = new_acc.sum().item() / new_acc.size(0)

    if it % 100 == 0:
        print("loss:", loss, "classification_token_loss", classification_token_loss, "reason_token_loss", reason_token_loss)
        print("new_acc", new_acc)


    return loss, discard_loss, classification_token_loss, reason_token_loss, new_acc, tokenizer.decode(first_token_labels)


def finetune(args, tokenizer: T5Tokenizer, model: deepspeed.DeepSpeedEngine, optimizer: Adafactor, dataset, device):
    print_rank("Start Fine-tuning") 

    # print_inspect(model, '*')
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        loss_func = mpu.parallel_cross_entropy
    # Standard Pytorch utilities
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)    

    sampler = DistributedSampler(dataset["train"], shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size)
    # Sampling, shuffling, batching (collate) & multiprocessing of data
    train_dataloader = DataLoader(
        dataset['train'], sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=dataset["train"].collate)

    step, global_step = 1, 1
    total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0



    if args.refinement:
        generation_config = GenerationConfig(
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_prompt_length,
        min_length=args.min_prompt_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Dict instead of logit tuple
        return_dict_in_generate=True,
        # Prediction Score
        output_scores=False,
        num_beams=3
        )
        og_labels =[]
            # Load each line
        with open(os.path.join(args.base_path, "data/anli/data.txt")) as f:
            raw_data = f.readlines()

        current_split = None
        for line in raw_data:
            # Remove any leading/trailing whitespace
            line = line.strip()  
            # Skip all train sections entirely
            if line.startswith("<dev_r1>") or line.startswith("<dev_r2>") or line.startswith("<dev_r3>"):
                current_split = None  # Set to None to ignore lines until next marker
                continue
            if line.startswith("<test_r1>") or line.startswith("<test_r2>") or line.startswith("<test_r3>"):
                current_split = None  # Set to None to ignore lines until next marker
                continue
            elif line.startswith("<train_r1>") or line.startswith("<train_r2>"):
                current_split = None
                continue
            elif line.startswith("<train_r3>"):
                current_split = True
                continue


            if current_split:
                try:
                    # Parse the line as JSON
                    data = json.loads(line.strip())
                    # Extract just the reason text value and add it to your labels
                    if "reason" in data:
                        # Check i reason has actually content
                        if len(data["reason"]) > 3:
                            og_labels.append(data["reason"].lower())
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line}")

        print("List of original labels: ", og_labels[:20])


    else:
        generation_config = GenerationConfig(
        # Sampling from models prob distr instead of greedy (most likely token)
        do_sample=args.do_sample,
        # Constrain sampling pool
        # Smallest # of tokens cummulative exceeding probability p
        top_p=args.top_p,
        # Most likely k tokens for sampling
        top_k=args.top_k,
        # Modulates next token probability
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_prompt_length,
        min_length=args.min_prompt_length,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Dict instead of logit tuple
        return_dict_in_generate=True,
        # Prediction Score
        output_scores=False
    )
        

    for epoch in range(args.epochs):
        # Compute loss (and metrics) before training
        evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device)
        sampler.set_epoch(epoch)

        metrics = []
        class_check = []
        running_acc = []
        accuracies = []
        model.train()
        if args.refinement:
            model.eval()

        for it, (model_batch, no_model_batch, gen_data) in enumerate(train_dataloader):# Check for NaN/Inf in input data
            if check_input_batch(model_batch, it) or check_input_batch(no_model_batch, it):
                print(f"Invalid values in input at iteration {it}")
                continue

            dataset["train"].move_to_device(model_batch, no_model_batch, gen_data, device)
            torch.cuda.synchronize()
            st_time = time.time()

            model_inputs = {
                "input_ids": model_batch["input_ids"],
                "attention_mask": model_batch["attention_mask"],
                "labels": no_model_batch["labels"],
                }
            # Get statistics of your input
            print("Input range:", model_inputs["input_ids"].min().item(), model_inputs["input_ids"].max().item())
            #print("Input mean:", model_inputs["input_ids"].mean().item())
            print("Input NaN check:", torch.isnan(model_inputs["input_ids"]).any())
            print("Input Inf check:", torch.isinf(model_inputs["input_ids"]).any())
            
            # Check inputs
            for key, value in model_batch.items():
                if torch.isnan(value).any():
                    print(f"NaN found in input {key} at iteration {it}")

            # Refine ANLI dataset
            if args.refinement:
                

                gen_model_inputs = {
                        "input_ids": model_batch["input_ids"],  # Take first example
                        "attention_mask": model_batch["attention_mask"]  # Take first example
                    }
                
                gen_out = model.generate(
                        **gen_model_inputs,
                        generation_config=generation_config)

                
                decoded_outputs = tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
                decoded_inputs = tokenizer.batch_decode(model_batch["input_ids"], skip_special_tokens=True)
                decoded_uids = tokenizer.batch_decode(no_model_batch["uids"], skip_special_tokens=True)
                decoded_premises = tokenizer.batch_decode(no_model_batch["premises"], skip_special_tokens=True)
                decoded_hypotheses = tokenizer.batch_decode(no_model_batch["hypotheses"], skip_special_tokens=True)



                gen_labels = no_model_batch["labels"].clone()
             
                gen_labels[gen_labels == -100] = tokenizer.pad_token_id
                decoded_labels = tokenizer.batch_decode(gen_labels, skip_special_tokens=True)


                for i, generated_reason in enumerate(decoded_outputs):
                    # Keep original reason if it is provided
                    if len(decoded_labels[i]) >= 15:
                        parts = decoded_labels[i].split(":", 1)  # Split on first occurrence only
                        label = parts[0].strip()  # "entailment"
                        reason = parts[1].strip() if len(parts) > 1 else ""  # "The model 
                        print("split reason: ",reason)
                        if reason.lower() in og_labels:
                            # Create the entry to append
                            entry = {
                                "uid": decoded_uids[i],
                                "premise": decoded_premises[i],
                                "hypothesis": decoded_hypotheses[i],
                                "label": label,
                                "reason": reason
                            }
                        else:
                            # Create the entry to append
                            entry = {
                                "uid": decoded_uids[i],
                                "premise": decoded_premises[i],
                                "hypothesis": decoded_hypotheses[i],
                                "label": label,
                                "reason": generated_reason
                            }

                        if it % 400 == 0:
                            print("og label", decoded_labels[i])
                            print(f"Label {i}: '{label}', Reason: '{generated_reason}'")


                    # No reason provided, add model output to dataset
                    # Filtered out later
                    else:                    
                        # Create the entry to append
                        entry = {
                            "uid": decoded_uids[i],
                            "premise": decoded_premises[i],
                            "hypothesis": decoded_hypotheses[i],
                            "label": decoded_labels[i],
                            "reason": generated_reason
                        }
                        
                    # Append to file
                    # Make sure the directory exists
                    if not os.path.exists(os.path.join(args.base_path, f"train_{args.anli_round}")):
                        os.makedirs(os.path.join(args.base_path, f"train_{args.anli_round}"))

                    # Now write to a .txt file inside that directory
                    with open(os.path.join(args.base_path, f"train_{args.anli_round}", "refined_data.txt"), 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                

            
    
            ### LOSS COMPUTATION ###

            logits = model(**model_inputs, use_cache=False).logits
            # print("Predictions NaN:", torch.isnan(logits).any())
            # print("Predictions range:", logits.min().item(), logits.max().item())
            labels = model_inputs["labels"]
            _, seq_len = labels.shape
                        
            

            # Compute loss
            loss, discard_loss, classification_loss, reason_loss, new_acc, classes = compute_class_reas_loss(args, logits, labels, device, seq_len, loss_func, compute_acc=True, tokenizer=tokenizer, it=it)
            
            class_check.append(classes)
            running_acc.append(new_acc)
            
            if it == 100 or it == 200 or it == 400:
                print("running acc", sum(running_acc) / len(running_acc))
                flattened = [item for sublist in class_check for item in sublist]
                print(class_check)
                print("class_check", Counter(flattened))

            # Sequence Level Accuracy
            if it % 300 == 0:
                # Take just the first example from the batch
                gen_model_inputs = {
                    "input_ids": model_batch["input_ids"],  # Take first example
                    "attention_mask": model_batch["attention_mask"]  # Take first example
                }

                gen_out = model.generate(
                    **gen_model_inputs,
                    generation_config=generation_config)
                
                decoded_outputs = tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)
                acc_labels = labels.clone()
                acc_labels[acc_labels == -100] = tokenizer.pad_token_id
                decoded_labels = tokenizer.batch_decode(acc_labels, skip_special_tokens=True)
                
                # First words (classifications) of batch
                # Then for your labels too
                for i, item in enumerate(decoded_labels):
                    print(f"Label {i}: '{item}', Length: {len(item)}")

                acc_labels =[item.split(':')[0].lower().rstrip(':') for item in decoded_labels]
                print("class_labels", acc_labels, "\n")
                # Add this before the problematic line
                for i, item in enumerate(decoded_outputs):
                    print(f"Output {i}: '{item}', Length: {len(item)}")

             
                # extract first word from output, removing : and what follows
                acc_outputs = [item.lower().split(':', 1)[0].split()[0] for item in decoded_outputs]
                try:
                    acc_outputs = [item.lower().split(':', 1)[0].split()[0] for item in decoded_outputs]

                except:
                    print("empty model output, skipping iteration")
                    continue

                # Compare each pair and create a list of booleans
                sequence_level_acc = [label == output for label, output in zip(acc_labels, acc_outputs)]

                # Calculate the overall accuracy
                accuracy = sum(sequence_level_acc) / len(sequence_level_acc)
                accuracies.append(f"Epoch:{str(epoch)}, It:{str(it)}, Sequence_Acc:{str(accuracy)}, Token_Acc:{str(new_acc)}")
                print("###sequence Level Accuracy:###", accuracy)
              
            if torch.isnan(loss).any():
                print(f"NaN loss at epoch {epoch}, iteration {it}")
                # Print additional debugging info
                print(f"Loss value: {loss.item()}")
            
            model.backward(loss)
            bad_gradients = check_gradients(model)

            if not bad_gradients and not discard_loss:
                model.step()
            else:
                print(f"NaN/Inf gradients detected at epoch {epoch}, iteration {it}:Skipping optimizer step")
                print(f"Current loss: {loss.item()}, classification loss: {classification_loss.item()}, reason loss: {reason_loss.item()}")
                # Optionally save debug state
                # debug_state = {
                #     'iteration': it,
                #     'epoch': epoch,
                #     'loss': loss.item(),
                #     'model_batch': model_batch,
                #     'no_model_batch': no_model_batch
                # }
                # torch.save(debug_state, f'debug_state_ep{epoch}_it{it}.pt')
            
            # Aggregate loss across all processes
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            global_loss = loss.mean().item() / dp_world_size

            global_distil_loss = 0

            torch.cuda.synchronize()
            elapsed_time = time.time() - st_time

            total_loss += global_loss
            total_time += elapsed_time

            # Logging
            def get_log(log_loss, log_distil_loss, log_time):
                return "train | epoch {:3d} | Iter: {:6d}/{:6d} | global iter: {:6d}/{:6d} | loss: {:.4f} | ds_loss: {:.4f} | lr: {:.4e} | scale: {:10.4f} | micro time: {:.3f} | step time: {:.3f}".format(
                    epoch,
                    step,
                    args.total_iters * args.gradient_accumulation_steps,
                    global_step,
                    args.total_iters,
                    log_loss,
                    log_distil_loss,
                    #lr_scheduler.get_last_lr()[0],
                    args.lr,
                    optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    elapsed_time,
                    log_time,
                )
                
            # Intermediate logging
            if args.mid_log_num > 0:
                mid_log_step = args.gradient_accumulation_steps // args.mid_log_num
                mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                if step % mid_log_step == 0:
                    print_rank(get_log(global_loss, global_distil_loss, 0))

            # Logs according to log interval and after backpropagation
            if global_step % args.log_interval == 0:
                print("args.save", args.save)       
                step_metrics= {
                    'epoch': epoch,
                    'step': step,
                    'total_steps': args.total_iters * args.gradient_accumulation_steps,
                    'global_step': global_step,
                    'total_iters': args.total_iters,
                    #'loss': total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    'loss': global_loss,
                    'classification_loss': classification_loss.cpu().item(),
                    'reason_loss': reason_loss.cpu().item(),
                    #'learning_rate': lr_scheduler.get_last_lr()[0],
                    'accuracy': 0,
                    #'scale': optimizer.cur_scale if hasattr(optimizer, "cur_scale") else 0,
                    'new_accuracy': new_acc,
                    'elapsed_time': elapsed_time,
                    'step_time': total_time / (args.log_interval)
                }
                metrics.append(step_metrics)


                fig = get_plot(args, metrics)
                p = os.path.join(args.save,f"step{global_step}_plot.png")
                fig.savefig(p)
                fig.show()


                log_str = get_log(
                    total_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_distil_loss / (args.log_interval * args.gradient_accumulation_steps),
                    total_time / (args.log_interval))
                print_rank("*" * 100)
                print_rank(log_str)
                print_rank(args.save)
                print_rank("*" * 100)
                save_rank(log_str, os.path.join(args.save, "log.txt"))
                # Reset for next step
                total_loss, total_distil_loss, total_time = 0.0, 0.0, 0.0
                        
            
            # Checkpointing  
            if epoch == 0 and step == 1:
                args.save_interval = 200
            
            if epoch == 0 and step == 202:
                args.save_interval = 3000


            # Increase checkpointing for later epochs 
            if epoch == 2 and step == 200:
                args.save_interval = 1000

            # Save path & and interval given
            if step % args.save_interval == 0:
                print("checkpointing accessed")
                save_dir_path = os.path.join(args.save, str(global_step))
                if args.model_parallel:
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        model.module.config.to_json_file(os.path.join(save_dir_path, "config.json"))
                        tokenizer.save_pretrained(save_dir_path)
                    if mpu.get_data_parallel_rank() == 0:
                        save_parallel(model.module, save_dir_path)
                else:
                    print("Saving model rank", dist.get_rank())
                    if dist.get_rank() == 0:
                        os.makedirs(save_dir_path, exist_ok=True)
                        print_rank(f"Model saved to {save_dir_path}")
                        tokenizer.save_pretrained(save_dir_path)
                        model.module.save_pretrained(save_dir_path, safe_serialization=False)
                dist.barrier()


            try:
                # Read the existing file
                with open(os.path.join(args.save,'Accuracies.json', 'r')) as f:
                    data = json.load(f)
                

                if isinstance(data, list):
                    data.append(accuracies)
                    data.append("\n")
    
                
                # Write back to the file
                with open('data.json', 'w') as f:
                    json.dump(data, f, indent=4)
                        
            except FileNotFoundError:
                # File doesn't exist, create it with list
                with open('Accuracies.json', 'w') as f:
                    json.dump(accuracies, f, indent=4)

            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0:
                evaluate(args, tokenizer, model, dataset["dev"], "dev", epoch, device)
                    
                model.train()
            
            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1
            
            if global_step > args.total_iters:
                break
            
    return model


def evaluate(args, tokenizer, model, dataset: LMTrainDataset, split, epoch, device):
    """
    Evaluation process. 
    If args.eval_gen is False, only the loss over batches is calculated.
    If args.eval_gen is True, the model generates responses and computes metrics.
    """

    # Handles transformation (dimensions, padding...) of samples into batches
    collate_fn = dataset.collate
    
    # Information regarding DistrC setup
    if args.model_parallel:
        dp_world_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()
        dp_group = mpu.get_data_parallel_group()
        loss_func = mpu.parallel_cross_entropy
    else:
        dp_world_size = dist.get_world_size()
        dp_rank = dist.get_rank()
        dp_group = None
        loss_func = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    print_rank("dp size", dp_world_size)

    generation_config = GenerationConfig(
        # Sampling from models prob distr instead of greedy (most likely token)
        do_sample=args.do_sample,
        # Constrain sampling pool
        # Smallest # of tokens cummulative exceeding probability p
        top_p=args.top_p,
        # Most likely k tokens for sampling
        top_k=args.top_k,
        # Modulates next token probability
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        min_length=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # Dict instead of logit tuple
        return_dict_in_generate=True,
        # Prediction Score
        output_scores=False
    )

    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False, rank=dp_rank, num_replicas=dp_world_size)
    dataloader = DataLoader(
        dataset, sampler=sampler, batch_size=args.eval_batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Inference mode (no dropout, batchnorm...)
    model.eval()
    all_loss = 0.0
    # Count processed batches
    step = 0

    # Accuracy metrics
    total_correct = 0
    total_samples = 0
    
    
    all_labels_ids = []
    class_check = []
    
    print("Start Evaluation")
    with torch.no_grad():
        for it, (model_batch, no_model_batch, gen_data) in enumerate(tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))):
            print_rank(f"{it}/{len(dataloader)}")
            
            dataset.move_to_device(model_batch, no_model_batch, gen_data, device)
            
            model_inputs = {
                "input_ids": model_batch["input_ids"],
                "attention_mask": model_batch["attention_mask"],
                "labels": no_model_batch["labels"]
                }
            
            model_outputs = model(**model_inputs)
            
            # Sequence Level Accuracy
            if it % 5 == 0:
                # Take just the first example from the batch
                gen_model_inputs = {
                    "input_ids": model_batch["input_ids"],  # Take first example
                    "attention_mask": model_batch["attention_mask"]  # Take first example
                }

                gen_out = model.generate(
                    **gen_model_inputs,
                    generation_config=generation_config)
            
                decoded_outputs = tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True)

                acc_labels = no_model_batch["labels"].clone()
                acc_labels[acc_labels == -100] = tokenizer.pad_token_id
                print("acc_labels eval", acc_labels, acc_labels.size)
                decoded_labels = tokenizer.batch_decode(acc_labels, skip_special_tokens=True)
            
                # First words (classifications) of batch
                acc_labels =[item.split(':')[0].lower().rstrip(':') for item in decoded_labels]
                try:
                    acc_outputs = [item.lower().split(':', 1)[0].split()[0] for item in decoded_outputs]

                except:
                    print("empty model output, skipping iteration")
                    continue

                # Compare each pair and create a list of booleans
                sequence_level_acc = [label == output for label, output in zip(acc_labels, acc_outputs)]
                
                # Calculate the overall accuracy
                accuracy = sum(sequence_level_acc) / len(sequence_level_acc)
                print("###sequence Level Accuracy eval:###", accuracy)
                args.sequence_level_acc[0].append(it)
                args.sequence_level_acc[1].append(float(accuracy))
                args.sequence_level_acc[2].append(float(accuracy))
                
                if it % 20 == 0:
                    # Format and print the results
                    print(f"{'Model Input:':<20} {tokenizer.decode(model_batch['input_ids'][0], skip_special_tokens=True)}")
                    print(f"{'Model Output:':<20} {decoded_outputs[0]}")  # Print first output
                    print(f"{'Label:':<20} {decoded_labels[0]}")  # Print first label

        

            
            # Pass raw predictions as keyword arguments (from dict)
            logits = model_outputs.logits
           
            labels = no_model_batch["labels"]
                
            _, seq_len = labels.shape

            # Compute loss
            loss,_, _, _, new_acc, classes = compute_class_reas_loss(args, logits, labels, device, seq_len, loss_func, compute_acc=True, tokenizer=tokenizer, it=it)
            
            
            class_check.append(classes)
            
            if it == 100:
                flattened = [item for sublist in class_check for item in sublist]
                print(class_check)
                print("class_check_eval", Counter(flattened))

            # Optionally print running accuracy
            if it % 20 == 0:
                print(f"token level accuraccy in step {it} is {new_acc:.4f}")
                print(f"Sequence level accuraccy in step {it} is {new_acc:.4f}")

            max_new_tokens = args.max_length - gen_data["input_ids"].size(1)
            
            # Generation
            if args.eval_gen:            
                gen_out = model.generate(
                    **gen_data,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens)
                
                full_ids = gen_out.sequences
                
                full_ids = F.pad(
                    full_ids,
                    (0, args.max_length - full_ids.shape[1]),
                    value=tokenizer.pad_token_id,
                )
                
                labels_ids = full_ids[:, gen_data["input_ids"].size(1):]
                all_labels_ids.append(labels_ids)
                
            # Sum of losses across devices
            dist.all_reduce(loss, dist.ReduceOp.SUM, group=dp_group)
            # Average loss across devices
            loss = loss / dp_world_size
            all_loss += loss.item()
            step += 1
    
    # Plot Accuracy
    plt.clf()  # Clear any existing plots
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(args.sequence_level_acc[0], args.sequence_level_acc[1], color='green')
    ax.set_title('Sequence Level Accuracy')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    save_dir_path = os.path.join(args.save, str(epoch))
    os.makedirs(save_dir_path, exist_ok=True)
    fig.savefig(os.path.join(save_dir_path, f"Seq_lev_acc_epoch-{epoch}.png"))
    plt.close(fig)
    args.sequence_level_acc[0] =[]
    args.sequence_level_acc[1] =[]
    # Plot complete sequence level accuracy
    plt.clf()  # Clear any existing plots
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.plot(range(len(args.sequence_level_acc[2])), args.sequence_level_acc[2], color='green')
    ax.set_title('Complete Sequence Level Accuracy')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(save_dir_path, f"Complete_Seq_lev_acc_epoch-{epoch}{epoch}.png"))
    plt.close(fig)

    
    if args.eval_gen:
        all_labels_ids = torch.cat(all_labels_ids, dim=0)
        # Combine model output from all devices (distr)
        all_labels_ids = all_gather(all_labels_ids, dim=1, world_size=dp_world_size, group=dp_group, op="stack")
        # flatten all but last dimension to unify gathered tensors
        all_labels_ids = all_labels_ids.view(-1, all_labels_ids.size(-1))
        
        labels = tokenizer.batch_decode(all_labels_ids, skip_special_tokens=True)
    
    if get_rank() == 0:
        if args.eval_gen:
            # Labels
            references = dataset.answers
            # Adjust length of model output
            labels = labels[:len(references)]
            
            res = compute_metrics(labels, references)
        
            eval_dir = os.path.join(args.save, "eval", str(epoch))
            print_rank(eval_dir)
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "answers.jsonl"), "w") as f:
                for resp in labels:
                    f.write(json.dumps({"text": resp}) + "\n")
        else:
            res = {}
    
        avg_loss = all_loss / step
        
        log_str = f"{split} | avg_loss: {avg_loss} | {res}"
        print_rank(log_str)
        save_rank(log_str, os.path.join(args.save, "log.txt"))


def main():
    torch.backends.cudnn.enabled = False
    
    args = get_args()
    initialize(args)

    if args.model_parallel:
        print("***running distributed training***")

    print("*** Default peft value:", args.peft, " ***")
    
    if dist.get_rank() == 0:
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)
    
    device = torch.cuda.current_device()
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_rank("\n\n" + "="*30 + f" EXP at {cur_time} " + "="*30, os.path.join(args.save, "log.txt"))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["gradient_clipping"] = args.clip_grad
    ds_config["steps_per_print"] = 10000000
    
    # ZeRo only needed for training
    if not args.do_train:
        ds_config["zero_optimization"]["stage"] = 0
    
    # Reduce floating point precision to 16 by disabling fp32
    args.fp32 = not ds_config["fp16"]["enabled"]    
    args.deepspeed_config = None
    
    # Get the tokenizer
    tokenizer = get_tokenizer(args)

    # Load dataset, obtain single samples using def finetune/collate
    dataset = prepare_dataset(
        args,
        tokenizer,
    )
    
    dp_world_size = mpu.get_data_parallel_world_size() if args.model_parallel else dist.get_world_size()
    
    if args.do_train:
        # One epoch: total numSamples / effective batch size (per gpu, number of dp units/gpu's, gradient accumulation steps) 
        args.train_iters_per_epoch = int(len(dataset["train"]) / (args.batch_size * dp_world_size * args.gradient_accumulation_steps))
        print_rank("Train iters per epoch", args.train_iters_per_epoch)
        # Recreate possible missing values
        if args.total_iters is None:
            args.total_iters = args.train_iters_per_epoch * args.epochs
        if args.epochs is None:
            args.epochs = math.ceil(args.total_iters / args.train_iters_per_epoch)
        print_rank("total_iters", args.total_iters)
        
        # Save model every epoch
        if args.save_interval == -1:
            args.save_interval = args.train_iters_per_epoch 
        # Evaluate every epoch
        if args.eval_interval == -1:
            args.eval_interval = args.train_iters_per_epoch
    
    #model, optimizer, lr_scheduler = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    model, optimizer = setup_model_and_optimizer(args, ds_config, device, set_optim=args.do_train)
    
    
    if args.do_train:
        #model = finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, device, teacher_model=teacher_model)
        model = finetune(args, tokenizer, model, optimizer, dataset, device)
   
    if args.do_eval:
        evaluate(args, tokenizer, model, dataset["test"], "test", 0, device)
        
    
if __name__ == "__main__":
    main()
