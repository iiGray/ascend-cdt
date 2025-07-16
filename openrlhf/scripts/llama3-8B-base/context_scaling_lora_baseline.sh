# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export HF_DATASETS_CACHE="/mnt/hwfile/opendatalab/tangzecheng/cache"
SAVE_DIR='/mnt/petrelfs/tangzecheng/local_ckpt'

deepspeed --master_port 29500 --include localhost:4,5,6,7 cli/context_denoise/baseline_language_modeling.py \
   --max_len 81920 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/pg19' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --pretrain 'meta-llama/Meta-Llama-3-8B' \
   --save_path ${SAVE_DIR}/pg19/Llama-3-8B-Scaling-CE/lora \
   --ckpt_path ${SAVE_DIR}/pg19/Llama-3-8B-Scaling-CE/lora  \
   --save_steps 50 \
   --lora_rank 64 \
   --logging_steps 1 \
   --eval_steps 25 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --num_training_samples 10000 \
   --max_epochs 2 \
   --input_key "text" \
   --pretrain_mode \
   --packing_samples \
   --bf16 \
   --num_processors 12 \
   --rope_theta 2e8 \
   --scaling_time 10 \
   --learning_rate 1e-6 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'context-scaling' \
   --wandb_run_name 'Llama-3-8B-Base-lora_32-ce' \
   --ring_attn_size 2 \
   --ring_head_stride 1;