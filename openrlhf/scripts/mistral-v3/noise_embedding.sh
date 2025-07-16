# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export HF_DATASETS_CACHE="/mnt/hwfile/opendatalab/tangzecheng/cache"
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed cli/context_denoise/train_sft.py \
   --max_len 80000 \
   --dataset 'ZetangForward/Long-context-training-V4' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --pretrain 'mistralai/Mistral-7B-Instruct-v0.3' \
   --save_path ${SAVE_DIR}/long-context-training-V4/Mistral-7B-Instruct-v0.3/full_v1 \
   --ckpt_path ${SAVE_DIR}/long-context-training-V4/Mistral-7B-Instruct-v0.3/full_v1  \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 25 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --input_key "message" \
   --packing_samples \
   --perturb_type "embedding" \
   --bf16 \
   --direction "other" \
   --num_processors 16 \
   --learning_rate 2e-6 \
   --flash_attn \
   --apply_chat_template \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --huggingface_key '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' \
   --wandb_project 'long-context-training-V4' \
   --wandb_run_name 'Mistral-7B-Instruct-v0.3_v4-embedding-epsilon_lr_2e-5' \
   --ring_attn_size 4 \
   --ring_head_stride 2;