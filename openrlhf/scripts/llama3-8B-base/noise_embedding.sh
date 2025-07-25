# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export HF_DATASETS_CACHE="/mnt/hwfile/opendatalab/tangzecheng/cache"
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --master_port 29500 --include localhost:0,1,2,3,4,5,6,7 cli/context_denoise/train_sft.py \
   --max_len 128000 \
   --dataset 'ZetangForward/Long-context-training-V1' \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
   --save_path ${SAVE_DIR}/long-context-training-V1/Llama-3.1-8B-Instruct/full \
   --ckpt_path ${SAVE_DIR}/long-context-training-V1/Llama-3.1-8B-Instruct/full  \
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
   --num_processors 16 \
   --learning_rate 1e-6 \
   --flash_attn \
   --apply_chat_template \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb '' \
   --wandb_project 'long-context-training-V1' \
   --wandb_run_name 'Llama-3.1-8B-Instruct-full-embedding-epsilon_lr' \
   --ring_attn_size 4 \
   --ring_head_stride 2;