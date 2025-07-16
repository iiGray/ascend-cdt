# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
SAVE_DIR='/data/zecheng/ckpt'

deepspeed --master_port 29500 --include localhost:0,1,2,3,4,5,6,7 cli/context_denoise/train_sft.py \
   --max_len 128000 \
   --dataset '/data/pub_data/ZetangForward/Long-context-training-V2' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --pretrain '/data/hf_models/Mistral-7B-Instruct-v0.3' \
   --save_path ${SAVE_DIR}/long-context-training-V2/Mistral-7B-Instruct-v0.3/full_v1 \
   --ckpt_path ${SAVE_DIR}/long-context-training-V2/Mistral-7B-Instruct-v0.3/full_v1  \
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
   --direction "opposite" \
   --num_processors 16 \
   --learning_rate 2e-6 \
   --flash_attn \
   --apply_chat_template \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --huggingface_key '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' \
   --wandb_project 'long-context-training-V2' \
   --wandb_run_name 'Mistral-7B-Instruct-v0.3-embedding-epsilon_lr_2e-6' \
   --ring_attn_size 4 \
   --ring_head_stride 2;