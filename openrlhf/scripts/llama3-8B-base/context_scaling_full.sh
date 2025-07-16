# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --master_port 29500 --include localhost:0,1,2,3,4,5,6,7 cli/context_denoise/context_scaling.py \
   --max_len 64000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/pg19' \
   --train_batch_size 32 \
   --micro_train_batch_size 1 \
   --pretrain '/mnt/hwfile/opendatalab/tangzecheng/local_ckpt/baseline/llama3-8B-pg19-longce/200ep' \
   --save_path ${SAVE_DIR}/pg19/Llama-3-8B-Scaling-Noise/full_v5 \
   --ckpt_path ${SAVE_DIR}/pg19/Llama-3-8B-Scaling-Noise/full_v5  \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 25 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --num_training_samples 15000 \
   --max_epochs 2 \
   --input_key "text" \
   --pretrain_mode \
   --packing_samples \
   --perturb_type "embedding" \
   --direction "other" \
   --bf16 \
   --num_processors 16 \
   --rope_theta 2e8 \
   --scaling_time 2 \
   --learning_rate 2e-5 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'context-scaling' \
   --wandb_run_name 'Llama-3-8B-Base-full_v5-embedding-epsilon_lr2e-5' \
   --ring_attn_size 8 \
   --ring_head_stride 2;