import math
import os
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from flash_attn.utils.distributed import all_gather
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
from tqdm import tqdm
from openrlhf.models import SimPOLoss
from openrlhf.utils.distributed_sampler import DistributedSampler
from openrlhf.models import GPTLMLoss


class CDTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        optim_kwargs (dict, defaults to {'lr':1e-4}): the kwargs to use while initializing optimizer
    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        pretrain_mode: bool = False,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        neg_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.pretrain_mode = pretrain_mode
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.neg_loss_weight = neg_loss_weight
        self.args = strategy.args
        
        self.loss_fn = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)

        # packing samples
        self.packing_samples = strategy.args.packing_samples

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("step_log/mini_batch_step")
            wandb.define_metric("step_log/*", step_metric="step_log/mini_batch_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

         # Restore step and start_epoch
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            pos_loss_mean, neg_loss_mean = 0, 0
            clue_pos_mean = 0

            for data in self.train_dataloader:  # here is the packing data since `flash_ring_attention` is applied

                prompt_id_lens, inputs, attention_masks, neg_inputs, neg_attention_masks, infos, clue_prompt_id_lens, clue_inputs, clue_attention_masks = data

                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                neg_inputs = neg_inputs.to(torch.cuda.current_device())
                neg_attention_masks = neg_attention_masks.to(torch.cuda.current_device())
                # if clue_inputs:
                clue_inputs = clue_inputs.to(torch.cuda.current_device()).squeeze(1)
                clue_attention_mask = clue_attention_masks.to(torch.cuda.current_device()).squeeze(1)

                # print(f"\ninputs shape is {inputs.shape}")  # FIXME
                # print(f"neg_inputs shape is {neg_inputs.shape}")  # FIXME

                pos_output = self.model(
                    inputs, 
                    attention_mask=attention_mask, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["input_length"],
                    cd_noise_settings={"add_noise": False},
                )

                
                # loss function
                pos_labels = torch.where(attention_mask.bool(), inputs, self.loss_fn.IGNORE_INDEX)
                # print(f"real input length of pos_labels: {attention_mask.sum()}")
                
                if not self.pretrain_mode:
                    index = 0
                    for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                        pos_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                        index += input_length

                pos_loss = self.loss_fn(pos_output.logits, pos_labels)
                self.strategy.backward(pos_loss, self.model, self.optimizer)
                
                neg_output = self.model(
                    neg_inputs, 
                    attention_mask=neg_attention_masks, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["neg_input_length"],
                    cd_noise_settings={"add_noise": True},
                )

                neg_labels = torch.where(neg_attention_masks.bool(), neg_inputs, self.loss_fn.IGNORE_INDEX)
                # print(f"real input length of neg_labels: {neg_attention_masks.sum()}")

                if not self.pretrain_mode:
                    index = 0
                    for input_length, source_len in zip(infos["neg_input_length"], prompt_id_lens):
                        neg_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                        index += input_length

                neg_loss = self.neg_loss_weight * self.loss_fn(neg_output.logits, neg_labels)
                # total_loss = pos_loss + self.neg_loss_weight * neg_loss

                # print(f"pos_loss: {pos_loss}, neg_loss: {neg_loss}")

                self.strategy.backward(neg_loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                pos_loss_mean = pos_loss_mean * 0.9 + 0.1 * pos_loss.item()
                neg_loss_mean = neg_loss_mean * 0.9 + 0.1 * neg_loss.item()

                ### ============================= ###
                # zecheng_note: 这里加上contextual 的labels，专门计算上下文的loss
                # with torch.no_grad():
                #     clue_output = self.model(
                #         clue_inputs, 
                #         attention_mask=clue_attention_mask, 
                #         return_output=True,
                #         ring_attn_group=self.strategy.ring_attn_group,
                #         packed_seq_lens=infos["clue_input_length"],
                #         cd_noise_settings={"add_noise": False},
                #     )

                #     clue_labels = torch.where(
                #         clue_attention_mask.bool(),
                #         clue_inputs,
                #         self.loss_fn.IGNORE_INDEX,
                #     )

                #     index = 0
                #     for input_length, source_len in zip(infos["clue_input_length"], clue_prompt_id_lens):
                #         clue_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                #         index += input_length

                #     short_ctx_gpt_loss = self.loss_fn(clue_output.logits, clue_labels)
                #     short_ctx_loss_mean = short_ctx_loss_mean * 0.9 + 0.1 * short_ctx_gpt_loss.item()

                logs_dict = {
                    # "short_ctx_loss": short_ctx_gpt_loss.item(),
                    "pos_loss_mean": pos_loss_mean,
                    "neg_loss_mean": neg_loss_mean,
                    "pos_loss": pos_loss.item(),
                    "neg_loss": neg_loss.item(),
                    # "short_ctx_loss_mean": short_ctx_loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }

                # step bar
                logs_dict = self.strategy.all_reduce(logs_dict)
                if self._wandb is not None and self.strategy.is_rank_0():
                    logs = {"step_log/%s" % k: v for k, v in {**logs_dict, "mini_batch_step": step}.items()}
                    self._wandb.log(logs)
                
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                # logs/checkpoints/evaluation
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)

                step += 1
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        # logs
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # eval
        if global_step % args.eval_steps == 0:
            self.evaluate(self.eval_dataloader, global_step)
        
        # save ckpt
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            # self.strategy.save_ckpt(
            #     self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
            # )
            self.strategy.save_model(self.model, self.tokenizer, os.path.join(args.save_path, tag))

    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            short_ctx_loss_sum = 0
            neg_loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )
 
            for data in eval_dataloader:
                prompt_id_lens, inputs, attention_masks, neg_inputs, neg_attention_masks, infos, clue_prompt_id_lens, clue_inputs, clue_attention_masks = data


                inputs = inputs.to(torch.cuda.current_device())
                attention_mask = attention_masks.to(torch.cuda.current_device())
                neg_inputs = neg_inputs.to(torch.cuda.current_device())
                neg_attention_masks = neg_attention_masks.to(torch.cuda.current_device())
                clue_inputs = clue_inputs.to(torch.cuda.current_device()).squeeze(1)
                clue_attention_mask = clue_attention_masks.to(torch.cuda.current_device()).squeeze(1)


                output = self.model(
                    inputs, 
                    attention_mask=attention_mask, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["input_length"],
                    cd_noise_settings={"add_noise": False}
                )

                neg_output = self.model(
                    neg_inputs, 
                    attention_mask=neg_attention_masks, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["neg_input_length"],
                    cd_noise_settings={"add_noise": False}
                )

                clue_output = self.model(
                    clue_inputs, 
                    attention_mask=clue_attention_mask, 
                    return_output=True,
                    ring_attn_group=self.strategy.ring_attn_group,
                    packed_seq_lens=infos["clue_input_length"],
                    cd_noise_settings={"add_noise": False}
                )

                clue_labels = torch.where(clue_attention_mask.bool(), clue_inputs, self.loss_fn.IGNORE_INDEX)
                neg_labels = torch.where(neg_attention_masks.bool(), neg_inputs, self.loss_fn.IGNORE_INDEX)
                labels = torch.where(attention_mask.bool(), inputs, self.loss_fn.IGNORE_INDEX)

                index = 0
                for input_length, source_len in zip(infos["clue_input_length"], clue_prompt_id_lens):
                    clue_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                    index += input_length
                
                index = 0
                for input_length, source_len in zip(infos["neg_input_length"], prompt_id_lens):
                    neg_labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                    index += input_length

                index = 0
                for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                    labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                    index += input_length

                short_ctx_gpt_loss = self.loss_fn(clue_output.logits, clue_labels)
                short_ctx_loss_sum += short_ctx_gpt_loss.item()

                loss = self.loss_fn(output.logits, labels)
                loss_sum += loss.item()
                
                neg_loss = self.loss_fn(neg_output.logits, neg_labels)
                neg_loss_sum += neg_loss.item()

                times += 1
                
                bar_dict = {
                    "eval gpt_loss": loss_sum / times, 
                    "eval neg_loss": neg_loss_sum / times,
                    "eval short_ctx_gpt_loss": short_ctx_loss_sum / times
                }

                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
        
