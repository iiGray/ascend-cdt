from typing import Optional
import os
from functools import partial
import torch
from torch import Tensor
from megatron.training import get_args, get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training import get_timers
try:
    from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import set_post_process_flag
except ImportError:
    pass

from megatron.core.models.gpt import GPTModel
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank, generate_actual_seq_len

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from megatron.core.inference.contexts import BaseInferenceContext

from megatron.core.packed_seq_params import PackedSeqParams

from megatron.core.utils import WrappedTensor, deprecate_inference_params

IGNORE_INDEX = -100



class CDTGPTModel(GPTModel):
    def _preprocess_embedding(
        self,
        input_ids: Tensor,
        input_embeds: Tensor,
        position_ids: Tensor,
        decoder_input: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        """Preprocesses inputs for the transformer decoder.

        Applies embeddings to input tokens, or uses `decoder_input` from a previous
        pipeline stage. Also sets up rotary positional embeddings.
        """

        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        in_inference_mode = inference_context is not None and not self.training

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = input_embeds
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        rotary_pos_cos = None
        rotary_pos_sin = None
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            if in_inference_mode and self.config.flash_decode:
                assert (
                    inference_context.is_static_batching()
                ), "GPTModel currently only supports static inference batching."
                # Flash decoding uses precomputed cos and sin for RoPE
                rotary_pos_cos, rotary_pos_sin = self.rotary_pos_emb_cache.setdefault(
                    inference_context.max_sequence_length,
                    self.rotary_pos_emb.get_cos_sin(inference_context.max_sequence_length),
                )
            else:
                rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                    inference_context, self.decoder, decoder_input, self.config, packed_seq_params
                )
                rotary_pos_emb = self.rotary_pos_emb(
                    rotary_seq_len,
                    packed_seq=packed_seq_params is not None
                    and packed_seq_params.qkv_format == 'thd',
                )
        elif self.position_embedding_type == 'mrope' and not self.config.multi_latent_attention:
            if self.training or not self.config.flash_decode:
                rotary_pos_emb = self.rotary_pos_emb(position_ids, self.mrope_section)
            else:
                # Flash decoding uses precomputed cos and sin for RoPE
                raise NotImplementedError(
                    "Flash decoding uses precomputed cos and sin for RoPE, not implmented in "
                    "MultimodalRotaryEmbedding yet."
                )

        if (
            in_inference_mode
            and (self.config.enable_cuda_graph or self.config.flash_decode)
            and rotary_pos_cos is not None
            and inference_context.is_static_batching()
        ):
            current_batch_size = input_ids.shape[0]
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * current_batch_size,
                dtype=torch.int32,
                device=rotary_pos_cos.device,  # Co-locate this with the rotary tensors
            )
        else:
            sequence_len_offset = None

        # Wrap decoder_input to allow the decoder (TransformerBlock) to delete the
        # reference held by this caller function, enabling early garbage collection for
        # inference. Skip wrapping if decoder_input is logged after decoder completion.
        if in_inference_mode and not has_config_logger_enabled(self.config):
            decoder_input = WrappedTensor(decoder_input)

        return decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset


    def forward_embedding(
        self,
        input_ids: Tensor,
        input_embeds: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self._preprocess_embedding(
                input_ids = input_ids,
                input_embeds = input_embeds,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            )
        )

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            mtp_in_postprocess=self.mtp_process,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )

    def forward_embedding_step(self, 
                               input_embeds: Tensor):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = CDTTrainer.get_batch(
            data_iterator)
        +++看看能不能传进来的时候不需要修改train_step
        timers('batch-generator').stop()

        # if args.use_legacy_models:
        #     output_tensor = model(tokens, position_ids, attention_mask,
        #                           labels=labels)
        # else:
        output_tensor = self.forward_embedding(tokens, position_ids, attention_mask,
                                labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(self.loss_func, loss_mask)

class CDTTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        args = get_args()
        if args.reset_position_ids:
            keys += ['position_ids']
        data_type = torch.int64

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.variable_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
                return tokens, None, None, attention_mask, None
            else:
                # Broadcast data.
                data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                if args.reset_position_ids:
                    generate_actual_seq_len(data_b)
                    batch = {'attention_mask': None}
                else:
                    attention_mask_1d = data_b.get('attention_mask').long()
                    attention_mask = get_tune_attention_mask(attention_mask_1d)
                    batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None

        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)

        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        # ignored label -100
        loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)

        if get_args().spec is not None and args.spec[0] == "mindspeed_llm.tasks.models.spec.hunyuan_spec":
            input_ids = tokens
            pad_id = 127961

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
            attention_mask = input_ids.ne(pad_id)

            position_ids = None
            batch = {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        else:
            if args.reset_position_ids:
                position_ids = data_b.get('position_ids').long()
                generate_actual_seq_len(data_b)

                batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                }
                batch = get_batch_on_this_cp_rank(batch)
                batch['attention_mask'] = None
                batch['position_ids'] = position_ids
                return batch.values()

            attention_mask = get_tune_attention_mask(attention_mask_1d)
            position_ids = None
            batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    @staticmethod
    def loss_func(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            input_tensor (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses
        """
        args = get_args()
        loss_mask = input_tensor

        losses = output_tensor.float()
        loss_mask = loss_mask[..., 1:].view(-1).float()
        if args.context_parallel_size > 1:
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss = loss[0] / loss[1]
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])

        return loss * args.context_parallel_size, {'lm loss': averaged_loss[0]}

    def forward_step(self, data_iterator, model: GPTModel):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)
        timers('batch-generator').stop()

        # if args.use_legacy_models:
        #     output_tensor = model(tokens, position_ids, attention_mask,
        #                           labels=labels)
        # else:
        output_tensor = model(tokens, position_ids, attention_mask,
                                labels=labels, loss_mask=loss_mask)
        
        

        return output_tensor, partial(self.loss_func, loss_mask)


def forward_step_in_sft_with_dualpipe(data_iterator, model, extra_block_kwargs=None):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """

    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    set_post_process_flag(model.module.module.post_process)
    tokens, labels, loss_mask, attention_mask, position_ids = CDTTrainer.get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if extra_block_kwargs is not None:
        # excute forward backward overlaping
        output_tensor, model_graph, pp_comm_output = \
            model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                  extra_block_kwargs=extra_block_kwargs)
        return (output_tensor, model_graph, pp_comm_output), partial(CDTTrainer.loss_func, loss_mask)
    else:
        output_tensor, model_graph = model(
            tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        return (output_tensor, model_graph), partial(CDTTrainer.loss_func, loss_mask)