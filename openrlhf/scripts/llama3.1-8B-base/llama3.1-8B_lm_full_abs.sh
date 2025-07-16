# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export HF_DATASETS_CACHE="/mnt/hwfile/opendatalab/tangzecheng/cache"
SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --master_port 29504 --include localhost:0,1,2,3,4,5,6,7 cli/context_denoise/language_modeling.py \
   --max_len 32000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/pg19' \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain 'Crystalcareai/meta-llama-3.1-8b' \
   --save_path ${SAVE_DIR}/pg19/Llama-3.1-8B/cd_lm_full-0.005 \
   --ckpt_path ${SAVE_DIR}/pg19/Llama-3.1-8B/cd_lm_full-0.005 \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --num_training_samples 10000 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --pretrain_mode \
   --adv_epsilon 0.005 \
   --input_key "text" \
   --packing_samples \
   --bf16 \
   --num_processors 16 \
   --learning_rate 5e-6 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project '初号机' \
   --wandb_run_name 'Llama-3.1-8B-cd-lm_full-adv0.005' \
   --ring_attn_size 4 \
   --ring_head_stride 2;