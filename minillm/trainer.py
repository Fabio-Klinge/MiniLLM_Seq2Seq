import json
import os
import deepspeed
from time import time
from typing import Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import (
    T5Tokenizer,
    GenerationConfig,
    mpu)

from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

from .utils import (
    get_scheduler_class,
    get_log_probs,
    get_rev_kl,
    significant
)

from .model import (
    PPOModel
)

from .pipelines import PPOPipeline, LMPipeline


from .storages import PPORolloutStorage
from .losses import Loss

from .visualizer import TrainingVisualizer

from utils import print_rank, save_rank, get_rank, all_gather, save_parallel
from rouge_metric import compute_metrics


class PPOTrainer():
    """
    RL model trainer with an `accelerate` based backend
    """

    def __init__(self, args, tokenizer: T5Tokenizer, reward_fn, ds_config):
        self.args = args
        self.max_length = args.max_length
        self.max_prompt_length = args.max_prompt_length
        self.ds_config = ds_config
        self.reward_fn = reward_fn
        self.device = torch.cuda.current_device()

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        if args.model_parallel:
            self.dp_world_size = mpu.get_data_parallel_world_size()
            self.dp_rank = mpu.get_data_parallel_rank()
            self.dp_group = mpu.get_data_parallel_group()
        else:
            self.dp_world_size = dist.get_world_size()
            self.dp_rank = dist.get_rank()
            self.dp_group = None

        self.model = PPOModel(args, self.device)
        if args.model_parallel:
            if mpu.get_data_parallel_rank() == 0:
                print(' > number of parameters on model parallel rank {}: {}M'.format(
                    mpu.get_model_parallel_rank(),
                    int(sum([p.nelement() for p in self.model.parameters()]) / 1e6)), flush=True)
        else:
            if dist.get_rank() == 0:
                print(' > number of parameters: {}M'.format(
                    int(sum([p.nelement() for p in self.model.parameters()]) / 1e6)), flush=True)

        self.sampler = None
        self.teacher_model = None
        self.opt = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()
        self.model, self.opt, self.scheduler = self.setup_ds(self.model, self.opt, self.scheduler)
        
        self.tokenizer = tokenizer
        self.store = PPORolloutStorage(self.tokenizer.pad_token_id, self.args.seed_ppo + self.dp_rank)
        self.store.clear_history()
        
        self.losses = Loss(args, self)
        self.generate_kwargs = dict(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            max_length=args.max_length,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    def set_teacher_model(self, model):
        self.teacher_model = model

    def set_sampler(self, sampler):
        self.sampler = sampler

    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's TRLConfig
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            betas=[0.9, 0.95],
            eps=1.0e-8,
            weight_decay=1.0e-6
        )

        return optimizer

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        if self.args.scheduler_name == "constant_trm":
            scheduler = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=self.args.warmup_iters)
        elif self.args.scheduler_name == "cosine_trm":
            scheduler = get_cosine_schedule_with_warmup(self.opt, num_warmup_steps=self.args.warmup_iters, num_training_steps=self.args.total_iters)
        else:
            scheduler_class = get_scheduler_class(self.args.scheduler_name)
            scheduler = scheduler_class(self.opt, eta_min=self.args.lr_min, T_max=self.args.total_iters)
        
        return scheduler

    def setup_ds(self, model, optimizer=None, scheduler=None):
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=self.args,
            lr_scheduler=scheduler,
            mpu=mpu if self.args.model_parallel else None,
            config_params=self.ds_config
        )
        return model, optimizer, scheduler

    def add_eval_pipeline(self, eval_pipeline: PPOPipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def add_lm_pipeline(self, lm_pipeline: LMPipeline, eval_lm_pipeline: LMPipeline):
        self.lm_pipeline = lm_pipeline
        self.eval_lm_pipeline = eval_lm_pipeline

    def get_model_inputs(
        self,
        input_tensors,
        response_tensors,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # tokens = torch.cat((query_tensors, response_tensors), dim=1)[
        #     :, -self.max_length :
        #]

        # Check mask computation, read claude for that, ask if get_mask is approriate, compute_l... -> mask = ...
        # 
        #
        attention_mask = self.get_mask(input_tensors)
  
        batch = {
            "input_ids": input_tensors.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": response_tensors.contiguous()
        }
        
        # if self.args.model_type in ["gpt2"]:  
        #     # For a proper positional encoding in case of left padding
        #     position_ids = attention_mask.cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask.eq(0), 0)
        #     batch["position_ids"] = position_ids
        
        return batch

    def get_mask(self, tokens):
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long()
        )
        return attention_mask

    def forward_model(self, batch):
        outputs = self.model(
            **batch,
            return_dict=True,
            use_cache=False,
        )
        return outputs

    def compute_logits_and_log_probs(self, input_ids, response_ids, inf_mask=None, base="base", return_logprobs=True):
        batch = self.get_model_inputs(
            input_ids, response_ids
        )
        
        if base == "base":
            model_cls = self.model.module.forward
        elif base == "teacher":
            model_cls = self.teacher_model
        else:
            raise NotImplementedError
        
        
        outputs = model_cls(
            **batch,
            return_dict=True,
            use_cache=False
        )

        logits = outputs.logits
        logits = logits / self.args.temperature

        if inf_mask is not None:
            logits = logits.masked_fill(inf_mask, -float("inf"))

        mask = batch["attention_mask"]
                
        if return_logprobs:
            logprobs = get_log_probs(logits, response_ids, mask, inf_mask, model_parallel=self.args.model_parallel)
            return logits, logprobs

        return logits

    def train(self):
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """

        self.prepare_learning()
        self.iter_count = 1
        self.global_iter_count = 1
        self.nth_evaluation = 0

        self.evaluate()
    # # keep smol
    # def hidden_train(self):
        
        print_rank("Total Steps:", self.total_steps, "Data Epochs:", self.args.epochs)
        lm_epochs = 0        
        logging_stats = defaultdict(float)
        visualizer = TrainingVisualizer(log_dir=self.args.save)

        for training_epoch in range(self.args.training_epochs):
            for ppo_epoch in range(self.n_updates_per_batch):
                for it, batch in enumerate(self.train_dataloader):
                    if self.lm_pipeline is not None:
                        try:
                            lm_batch = next(self.lm_iterator)
                        except StopIteration:
                            lm_epochs += 1
                            print_rank(f"Another lm epoch, lm epochs: {lm_epochs}")
                            save_rank(f"Another lm epoch, lm epochs: {lm_epochs}", os.path.join(self.args.save, "log.txt"))
                            self.lm_dataloader.sampler.set_epoch(lm_epochs)
                            self.lm_iterator = iter(self.lm_dataloader)
                            lm_batch = next(self.lm_iterator)


                        self.lm_pipeline.move_to_device(*lm_batch, self.device)
                    self.store.move_to_device(batch, self.device)
                    stats = {}

                    if self.args.model_parallel:
                        self.store.broadcast(batch, src=mpu.get_model_parallel_src_rank(), group=mpu.get_model_parallel_group())

                    if self.args.gradient_checkpointing:
                        self.model.module.set_force_gradient_checkpointing(True)
                    
                    input_batch = self.losses.get_input_batch(batch, None)

                    # compute lm-related loss on pre-training data (optinal)
                    if self.lm_pipeline is not None:
                        pt_loss, pt_loss_stats = self.losses.pt_loss(lm_batch, lm_logits)
                        stats.update(pt_loss_stats)
                    else:
                        pt_loss = 0

                    logits = self.forward_model(input_batch).logits
                    ppo_logits = logits[:batch.query_tensors.size(0)]
                    lm_logits = logits[batch.query_tensors.size(0):]

                    # forward
                    forward_time = time()
                    # compute rl-related loss on explored data
                    rl_loss, rl_loss_stats = self.losses.ppo_loss(batch, ppo_logits)
                    stats.update(rl_loss_stats)
          
                    loss = rl_loss + self.args.lm_coef * pt_loss
                    stats["tot_loss"] = loss.item()

                    forward_time = time() - forward_time
                    
                    # backward
                    backward_time = time()
                    self.model.backward(loss)
                    backward_time = time() - backward_time

                    # step
                    step_time = time()
                    self.model.step()
                    step_time = time() - step_time

                    if self.args.gradient_checkpointing:
                        self.model.module.set_force_gradient_checkpointing(False)

                    if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                        ((self.global_iter_count < 10000 and (self.global_iter_count % 1000 == 0)) or \
                        self.global_iter_count % self.args.save_interval == 0):
                        self.save()

                    # eval
                    if self.iter_count % self.args.gradient_accumulation_steps == 0 and \
                        ((self.global_iter_count < 1000 and (self.global_iter_count % 100 == 0)) or \
                        (self.global_iter_count % self.args.eval_interval == 0)):
                        self.evaluate()

                    elapsed_time = forward_time + backward_time + step_time
                    
                    stats["elapsed_time"] = elapsed_time
                    
                    for k in stats:
                        logging_stats[k] += stats[k]

                    # #Logging
                    # def get_log_and_plot(log_stats, one_step_time, visualizer=None):
                    #     keys = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", "reward", "rev_kl", "stu_lens", "mixed_lens"]
                    #     prefix = "train | data_epochs {:2d}/{:2d} | inner iter: {:3d}/{:3d} | ppo epoch: {:2d}/{:2d} | global iter: {:6d}/{:6d}".format(
                    #         self.sampler.epochs,
                    #         self.args.epochs,
                    #         it,
                    #         len(self.train_dataloader),
                    #         ppo_epoch,
                    #         self.n_updates_per_batch,
                    #         self.global_iter_count,
                    #         self.total_steps
                    #     )
                    #     suffix = "| lr: {:.4e} | scale: {:6.2f} | time: {:.3f} | step time: {:.3f}".format(
                    #         self.scheduler.get_last_lr()[0],
                    #         self.opt.cur_scale if hasattr(self.opt, "cur_scale") else 0,
                    #         elapsed_time,
                    #         one_step_time
                    #     )
                    #     for key in keys:
                    #         prefix += "| {}: {:.4f} ".format(key, log_stats.get(key, 0))
                    #     return prefix + suffix


                    # def get_log_and_plot(log_stats, one_step_time, visualizer=None):
                    #     # Logging
                    #     keys = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", 
                    #             "reward", "rev_kl", "stu_lens", "mixed_lens"]
                    #     prefix = "train | data_epochs {:2d}/{:2d} | inner iter: {:3d}/{:3d} | ppo epoch: {:2d}/{:2d} | global iter: {:6d}/{:6d}".format(
                    #         self.sampler.epochs,
                    #         self.args.epochs,
                    #         it,
                    #         len(self.train_dataloader),
                    #         ppo_epoch,
                    #         self.n_updates_per_batch,
                    #         self.global_iter_count,
                    #         self.total_steps
                    #     )
                    #     suffix = "| lr: {:.4e} | scale: {:6.2f} | time: {:.3f} | step time: {:.3f}".format(
                    #         self.scheduler.get_last_lr()[0],
                    #         self.opt.cur_scale if hasattr(self.opt, "cur_scale") else 0,
                    #         elapsed_time,
                    #         one_step_time
                    #     )
                        
                    #     # Update visualizer if provided
                    #     if visualizer is not None:
                    #         try:
                    #             visualizer.update(plot_stats)
                                
                    #             # Plot every N steps (adjust as needed)
                    #             if self.global_iter_count % 100 == 0:
                    #                 plot_path = os.path.join(visualizer.log_dir, f'training_plot_step_{self.global_iter_count}.png')
                    #                 visualizer.plot(save_path=plot_path)
                    #                 print(f"Generated plot at step {self.global_iter_count}")
                    #         except Exception as e:
                    #             print(f"Error in visualization: {e}")
    
                        
                    #     return prefix + suffix
                    
                    def get_log_and_plot(log_stats, one_step_time, visualizer=None):
                        """Enhanced logging function with visualization"""
                        # Original logging logic
                        keys = ["tot_loss", "rl_loss", "pt_loss", "pg_loss", "reg_loss", 
                                "reward", "rev_kl", "stu_lens", "mixed_lens"]
                        
                        # Create a clean dictionary of just the numeric metrics
                        plot_stats = {}
                        for key in log_stats:
                            value = log_stats.get(key, 0)
                            if isinstance(value, (int, float)):
                                plot_stats[key] = value
                        
                        # Original string formatting
                        prefix = "train | data_epochs {:2d}/{:2d} | inner iter: {:3d}/{:3d} | ppo epoch: {:2d}/{:2d} | global iter: {:6d}/{:6d}".format(
                            self.sampler.epochs,
                            self.args.epochs,
                            it,
                            len(self.train_dataloader),
                            ppo_epoch,
                            self.n_updates_per_batch,
                            self.global_iter_count,
                            self.total_steps
                        )
                        
                        suffix = "| lr: {:.4e} | scale: {:6.2f} | time: {:.3f} | step time: {:.3f}".format(
                            self.scheduler.get_last_lr()[0],
                            self.opt.cur_scale if hasattr(self.opt, "cur_scale") else 0,
                            elapsed_time,
                            one_step_time
                        )
                        
                        # Add metrics to log string
                        for key in keys:
                            prefix += "| {}: {:.4f} ".format(key, log_stats.get(key, 0))
                        
                        # Update visualizer if provided
                        if visualizer is not None:
                            try:
                                visualizer.update(plot_stats)
                                
                                # Plot every N steps (adjust as needed)
                                if self.global_iter_count % 100 == 0:
                                    plot_path = os.path.join(visualizer.log_dir, f'training_plot_step_{self.global_iter_count}.png')
                                    visualizer.plot(save_path=plot_path)
                                    print(f"Generated plot at step {self.global_iter_count}")
                            except Exception as e:
                                print(f"Error in visualization: {e}")
                        
                        return prefix + suffix

                    

                    # Log and plot
                    mid_log_step = self.args.gradient_accumulation_steps // self.args.mid_log_num
                    mid_log_step = 1 if mid_log_step == 0 else mid_log_step
                    if self.iter_count % mid_log_step == 0:
                        print_rank(get_log_and_plot(stats, 0, visualizer))

                    if self.global_iter_count % self.args.log_interval == 0 and self.iter_count % self.args.gradient_accumulation_steps == 0:
                        logging_stats = {k:v/(self.args.log_interval*self.args.gradient_accumulation_steps) for k,v in logging_stats.items()}
                        log_str = get_log_and_plot(logging_stats, logging_stats.get("elapsed_time", 0) * self.args.gradient_accumulation_steps, visualizer)
                        print_rank("*" * 100)
                        print_rank(log_str)
                        print_rank(self.args.save)
                        print_rank("*" * 100)
                        save_rank(log_str, os.path.join(self.args.save, "log.txt"))
                        logging_stats = {k:0 for k in logging_stats}

                    # end
                    if (self.global_iter_count >= self.total_steps or self.sampler.epochs >= self.args.epochs):
                        if self.global_iter_count >= self.total_steps:
                            print_rank("Reached total steps {}/{}".format(self.global_iter_count, self.total_steps))
                        else:
                            print_rank("Reached data epochs {}/{}".format(self.sampler.epochs, self.args.epochs)) 
                        self.save()
                        results, preds, response_texts = self.evaluate_ppo()
                        if self.eval_lm_pipeline is not None:
                            eval_pt_results = self.evaluate_pt()
                            results.update(eval_pt_results)
                        self.save_evals(preds, results, response_texts)
                        return results
                    
                    self.iter_count += 1
                    if self.iter_count % self.args.gradient_accumulation_steps == 0:
                        self.global_iter_count += 1

                self.post_backward_callback()

            self.post_epoch_callback(training_epoch)

    def post_backward_callback(self):
        pass
        
    def post_epoch_callback(self, epoch):
        self.store.clear_history()
        # self.store.load(self.args.save)
        self.sampler.run_sample(
            self.args.num_rollouts_per_device, self.global_iter_count
        )  # Collect more rollouts for training

    def prepare_learning(self):
        self.lm_pipeline = None
        self.train_dataloader = self.store.create_loader(
            self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True
        )
        
        self.eval_dataloader = self.eval_pipeline.create_loader(
            self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=False)
        # Otional lm data
        if self.lm_pipeline is not None:
            self.lm_dataloader = self.lm_pipeline.create_loader(
                self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True)
            self.lm_iterator = iter(self.lm_dataloader)
        if self.lm_pipeline is not None:
            self.eval_lm_dataloader = self.eval_lm_pipeline.create_loader(
                self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=False)

        self.n_updates_per_batch = self.args.ppo_epochs
        self.total_steps = int(
            self.args.training_epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
            / self.args.gradient_accumulation_steps
        )
        self.total_steps = min(self.total_steps, self.args.total_iters)

    def evaluate(self):
        eval_results = {}
        eval_rl_results, preds, response_texts = self.evaluate_ppo()
        eval_results.update(eval_rl_results)
        if self.eval_lm_pipeline is not None:
            eval_pt_results = self.evaluate_pt()
            eval_results.update(eval_pt_results)
            
        response_texts = response_texts[:len(self.eval_pipeline.ppo_answers)]            
        self.save_evals(preds, eval_results, response_texts)
        
        if get_rank() == 0:
            res = compute_metrics(response_texts, self.eval_pipeline.ppo_answers)
            eval_results.update(res)
            keys = ["rougeL", "exact_match", "rev_kl", "lens"]
            eval_log_str = "eval "
            for key in keys:
                eval_log_str += "| {}: {:.3f} ".format(key, eval_results[key])
            print_rank(eval_log_str)
            save_rank(eval_log_str, os.path.join(self.args.save, "log.txt"))

    def evaluate_ppo(self):  # noqa: C901
        # self.model.eval()
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        stats = {}
        all_input_ids = []
        all_gen_ids = []
        all_rev_kl = []
        all_lens = []
        
        table = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, "Generation Evaluation", disable=(not get_rank() == 0)):
                batch, no_model_batch = batch
                batch, _ = self.eval_pipeline.move_to_device(batch, no_model_batch, self.device)
                gen_out = self.generate(
                    **batch,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                gen_ids = gen_out.sequences
                gen_logits = gen_out.scores # NOTE: [b, s, h_p]
                inf_mask = torch.isinf(gen_logits)
                
                input_ids = batch["input_ids"]
                #response_ids = batch["labels"]
                #gen_ids = full_ids[:, input_ids.size(1):]
                mask = self.get_mask(gen_ids)
                #mask = mask[:, input_ids.size(1)-1:input_ids.size(1)+gen_ids.size(1)-1]
                lens = torch.sum(mask, dim=-1)
                
                teacher_rewards = self.reward_fn(input_ids, gen_ids)["rewards"] # \log p(y_t | y_{<t}, x)
                _, logprobs = self.compute_logits_and_log_probs(input_ids, gen_ids, inf_mask=inf_mask, base="base") # \log q_{\theta}(y_t | y_{<t}, x)
                
                all_input_ids.append(input_ids)
                all_gen_ids.append(gen_ids)
                kl = get_rev_kl(teacher_rewards, logprobs, mask)
                kl = kl.sum(-1)
                
                if self.args.length_norm:
                    kl = kl / lens

                all_rev_kl.append(kl)
                all_lens.append(lens)
            #print("###all_input_ids (before cat)###", all_input_ids.shape)
            all_input_ids = torch.cat(all_input_ids, dim=0)
            all_gen_ids = torch.cat(all_gen_ids, dim=0)
            all_rev_kl = torch.cat(all_rev_kl, dim=0)
            all_lens = torch.cat(all_lens, dim=0)
            # print("###all_input_ids###", all_input_ids.shape)
            # print("###all_gen_ids###", all_gen_ids.shape)

            input_ids = all_gather(all_input_ids, dim=1, world_size=self.dp_world_size, group=self.dp_group, op="stack")
            gen_ids = all_gather(all_gen_ids, dim=1, world_size=self.dp_world_size, group=self.dp_group, op="stack")
            # Flatten extra dimension
            input_ids = input_ids.view(-1, input_ids.size(-1))
            gen_ids = gen_ids.view(-1, gen_ids.size(-1))
            print("evalppo")
            print(input_ids)
            print(input_ids.shape if isinstance(input_ids, torch.Tensor) else type(input_ids))
            print(gen_ids)
            print(gen_ids.shape if isinstance(gen_ids, torch.Tensor) else type(gen_ids))

            #prompt_ids = full_ids[:, :self.eval_pipeline.max_prompt_length]
            all_rev_kl = all_gather(all_rev_kl, dim=0, world_size=self.dp_world_size, group=self.dp_group)
            stats["rev_kl"] = all_rev_kl.mean()
            all_lens = all_gather(all_lens, dim=0, world_size=self.dp_world_size, group=self.dp_group)
            stats["lens"] = all_lens.float().mean()

            response_texts = []
            
            if get_rank() == 0:
                prompt_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                response_texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                print("decode works sometimes")
                gen_texts = [p + g for p, g in zip(prompt_texts, response_texts)]

                columns = ["prompts"]
                columns_data = [prompt_texts]
                # in online setting, compute the reward for validation
                columns.append("samples")
                if isinstance(gen_texts[0], str):
                    columns_data.append(gen_texts)
                else:
                    columns_data.append(gen_texts.tolist())

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        if get_rank() == 0:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)

            for ix in range(min(3, len(rows))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])

            try:
                Console().print(rich_table)
            except:
                pass

        self.nth_evaluation += 1
        return stats, table, response_texts

    def evaluate_pt(self):
        all_pt_losses = []
        all_lm_losses = []
        all_kd_losses = []
        for batch in tqdm(self.eval_lm_dataloader, desc="LM Evaluation", disable=(not get_rank() == 0)):
            self.eval_lm_pipeline.move_to_device(*batch, self.device)
            model_batch, _ = batch
            #print("evalpt", model_batch, type(model_batch))
            gen_out = self.generate(**model_batch, return_dict_in_generate=True, mode = "teacher", output_scores=True)
            response_ids = gen_out.sequences
            model_batch["labels"] = response_ids
            outputs = self.model(**model_batch, return_dict=True, use_cache=False)
            logits = outputs.logits
            with torch.no_grad():
                _, stats = self.losses.pt_loss(batch, logits)
                all_pt_losses.append(stats["pt_loss"])
                all_lm_losses.append(stats["lm_loss"])
                all_kd_losses.append(stats["ds_loss"])
        
        all_pt_losses = torch.tensor(all_pt_losses, device=self.device)
        eval_pt_loss = all_gather(all_pt_losses, dim=0, world_size=self.dp_world_size, group=self.dp_group).mean().item()
        
        all_lm_losses = torch.tensor(all_lm_losses, device=self.device)
        eval_lm_loss = all_gather(all_lm_losses, dim=0, world_size=self.dp_world_size, group=self.dp_group).mean().item()
        
        all_kd_losses = torch.tensor(all_kd_losses, device=self.device)
        eval_kd_loss = all_gather(all_kd_losses, dim=0, world_size=self.dp_world_size, group=self.dp_group).mean().item()
        
        results = {"pt_loss": eval_pt_loss, "lm_loss": eval_lm_loss, "kd_loss": eval_kd_loss}
        
        return results
    
    def save(self, directory: Optional[str] = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        ckpt_dir = os.path.join(base_ckpt_path, f"{self.global_iter_count}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if self.args.model_parallel:
            if get_rank() == 0:
                self.model.module.config.to_json_file(os.path.join(ckpt_dir, "config.json"))
                self.tokenizer.save_pretrained(ckpt_dir)
            if mpu.get_data_parallel_rank() == 0:
                save_parallel(self.model.module.base_model, ckpt_dir)
        else:
            if get_rank() == 0:
                self.model.module.base_model.save_pretrained(ckpt_dir, safe_serialization=False)
                # torch.save(self.model.module.value_model.state_dict(), os.path.join(ckpt_dir, "value_model.ckpt"))
                print(f"Model save to {ckpt_dir}")
                self.tokenizer.save_pretrained(ckpt_dir)

    def save_evals(self, preds, results, response_texts, directory: Optional[str] = None):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        """Creates checkpoint of optimizer, scheduler and a model"""
        base_ckpt_path = directory or self.args.save
        save_dir = os.path.join(base_ckpt_path, "eval", f"{self.global_iter_count}")
        os.makedirs(save_dir, exist_ok=True)
        
        if get_rank() == 0:
            torch.save(preds, os.path.join(save_dir, "preds.pt"))
            torch.save(results, os.path.join(save_dir, "results.pt"))
            with open(os.path.join(save_dir, "answers.jsonl"), "w") as f:
                for resp in response_texts:
                    f.write(json.dumps({"text": resp}) + "\n")

    def push_to_store(self, data):
        self.store.push(data)
         
    # gen_out = self.trainer.generate(**batch, return_dict_in_generate=True, mode=mode, teacher_mixed_sample=(self.args.teacher_mixed_alpha is not None), output_scores=True)
                
    # def generate(self, input_ids, attention_mask=None, mode="base", teacher_mixed_sample=False, **kwargs):
    #     """Wraps hf's `generate` adding some specific method's defaults"""
    #     input_ids = input_ids.to(self.device)
    #     if attention_mask is not None:
    #         attention_mask = attention_mask.to(self.device)

    #     kwargs = dict(self.generate_kwargs, **kwargs)

    #     if mode == "base":
    #         model = self.model.module
    #     elif mode == "teacher":
    #         model = self.teacher_model
    #     else:
    #         raise NotImplementedError

    #     mix_in_model, mix_in_alpha = None, None
    #     if teacher_mixed_sample:
    #         mix_in_model = self.teacher_model
    #         mix_in_alpha = self.args.teacher_mixed_alpha
        
    #     with torch.no_grad():
            
    #         generation_config = GenerationConfig(**kwargs)
    #         print("###Maxl in generate:", generation_config.max_length)
    #         max_new_tokens = min(10,generation_config.max_length - input_ids.size(1))
    #         gen = model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             generation_config=generation_config,
    #             max_new_tokens=max_new_tokens,
    #             mix_in_model=mix_in_model,
    #             mix_in_alpha=mix_in_alpha
    #         )

            
            
    #         gen.sequences = F.pad(
    #             gen.sequences,
    #             (0, self.max_length - gen.sequences.shape[1]),
    #             value=self.tokenizer.pad_token_id,
    #         )

    #         print("Check if paddinng worked and gen.sequences contains both input_ids&model output \nInput shape:", input_ids.shape)
    #         print("Generated shape:", gen.sequences.shape)
    #         print("Input text:", self.tokenizer.decode(input_ids[0]))
    #         print("Full generated text:", self.tokenizer.decode(gen.sequences[0]))
                        
    #         if gen.scores is not None:
    #             gen.scores = torch.stack(gen.scores, dim=1)
    #             gen.scores = torch.cat([
    #                 gen.scores, 
    #                 torch.zeros(
    #                     gen.scores.size(0),
    #                     self.max_length - self.args.max_prompt_length - gen.scores.size(1),
    #                     gen.scores.size(2),
    #                     device=gen.scores.device)],
    #                 dim=1)

    #         # Check the dimensions of gen.scores
    #         b, s, h_p = gen.scores.shape


    #         print(f"gen.scores shape: [{b}, {s}, {h_p}]")



    #         # Add an assertion to ensure the shape is correct
    #         assert gen.scores.shape[2] == model.config.vocab_size, f"Expected h_p to be {self.model.config.vocab_size}, but got {h_p}"
  
    #         # NOTE: scores: [b, s, h_p]

    #     return gen
        
    def generate(self, input_ids, attention_mask=None, mode="base", teacher_mixed_sample=False, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        kwargs = dict(self.generate_kwargs, **kwargs)
        if mode == "base":
            model = self.model.module
        elif mode == "teacher":
            model = self.teacher_model
        else:
            raise NotImplementedError
        mix_in_model, mix_in_alpha = None, None
        if teacher_mixed_sample:
            mix_in_model = self.teacher_model
            mix_in_alpha = self.args.teacher_mixed_alpha

        # print("\n--- Input IDs ---")
        # print(f"Shape: {input_ids.shape}")
        # print("Decoded input:")
        # for i, seq in enumerate(input_ids):
        #     print(f"  Sequence {i}: {self.tokenizer.decode(seq, skip_special_tokens=False)}")

        with torch.no_grad():
            generation_config = GenerationConfig(**kwargs)
            # print("\n--- Generation Config ---")
            # print(f"Max length: {generation_config.max_length}, input ids size: {input_ids.size(1)}")

            max_new_tokens = generation_config.max_length - input_ids.size(1)
            # print(f"Max new tokens: {max_new_tokens}")

            if self.teacher_model:
                
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    mix_in_model=mix_in_model,
                    mix_in_alpha=mix_in_alpha
                )

            # print("\n--- Generated Sequences (before padding) ---")
            # print(f"Shape: {gen.sequences.shape}")
            # print("Decoded sequences:")
            # for i, seq in enumerate(gen.sequences):
            #     print(f"  Sequence {i}: {self.tokenizer.decode(seq, skip_special_tokens=False)}")

            gen.sequences = F.pad(
                gen.sequences,
                (0, self.max_prompt_length - gen.sequences.shape[1]),
                value=self.tokenizer.pad_token_id,
            )

            # print("\n--- Generated Sequences (after padding) ---")
            # print(f"Shape: {gen.sequences.shape}")
            # print("Decoded sequences:")
            # for i, seq in enumerate(gen.sequences):
            #     print(f"  Sequence {i}: {self.tokenizer.decode(seq, skip_special_tokens=False)}")

            # print("\n--- Scores ---")
            # dimensions = [tensor.shape for tensor in gen.scores]
            # print(f"All dimensions: {dimensions}")

            #print("Dimension 1 size before concatenation:", gen.scores.size(1))
            scores_test = gen.scores

            if gen.scores is not None:
                gen.scores = torch.stack(gen.scores, dim=1)
                # print("stack shape:", gen.scores.shape if isinstance(gen.scores, torch.Tensor) else "Not a tensor")
                # print("for concat l:", self.max_length, "max prompt len", self.args.max_prompt_length,"1. dim scores size", gen.scores.size(1))
                gen.scores = torch.cat([
                    gen.scores,
                    torch.zeros(
                        gen.scores.size(0),
                        self.max_length - self.args.max_prompt_length - gen.scores.size(1),
                        gen.scores.size(2),
                        device=gen.scores.device)],
                    dim=1)
                
                # print("\n--- Scores ---")
                # print("Dimension 1 size after concatenation:", gen.scores.size(1))
                # print("score shape after concat:", gen.scores.shape if isinstance(gen.scores, torch.Tensor) else "Not a tensor")
               

                
            if gen.scores is not None:
                gen.scores = torch.stack(scores_test, dim=1)
                #print("for concat l:", self.max_length, "max prompt len", self.args.max_prompt_length,"1. dim scores size", gen.scores.size(1))
                gen.scores = torch.cat([
                    gen.scores,
                    torch.zeros(
                        gen.scores.size(0),
                        self.max_length - self.args.max_prompt_length - gen.scores.size(1),
                        gen.scores.size(2),
                        device=gen.scores.device)],
                    dim=1)
                

            # print("\n--- Final Check ---")
            # print(f"Input shape: {input_ids.shape}")
            # print(f"Generated shape: {gen.sequences.shape}")
            # print(f"Example input: {self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
            # print(f"Example full generated: {self.tokenizer.decode(gen.sequences[0], skip_special_tokens=False)}")

        return gen
