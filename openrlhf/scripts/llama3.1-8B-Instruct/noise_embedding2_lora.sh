# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export HF_DATASETS_CACHE="/mnt/hwfile/opendatalab/tangzecheng/cache"
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --master_port 29500 --include localhost:0,1,2,3,4,5,6,7 cli/context_denoise/train_sft.py \
   --max_len 80000 \
   --dataset 'ZetangForward/Long-context-training-V4' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --lora_rank 32 \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/long-context-training-V4/Llama-3.1-8B-Instruct/lora_v1 \
   --ckpt_path ${SAVE_DIR}/long-context-training-V4/Llama-3.1-8B-Instruct/lora_v1  \
   --save_steps 100 \
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
   --learning_rate 5e-6 \
   --flash_attn \
   --apply_chat_template \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --huggingface_key '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' \
   --wandb_project 'long-context-training-V4' \
   --wandb_run_name 'Llama-3.1-8B-Instruct-lora_v1-embedding-other-epsilon_lr_5e-6' \
   --ring_attn_size 4 \
   --ring_head_stride 4;