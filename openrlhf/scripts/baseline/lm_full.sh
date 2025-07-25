SAVE_DIR='/mnt/petrelfs/tangzecheng/remote_bucket/zecheng/ckpt'

deepspeed --include localhost:0,1,2,3,4,5,6,7 cli/baseline/baseline_language_modeling.py \
   --max_len 32000 \
   --dataset '/mnt/petrelfs/tangzecheng/local_data/pg19' \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain 'Crystalcareai/meta-llama-3.1-8b' \
   --save_path ${SAVE_DIR}/pg19/Llama-3.1-8B/lm_full_ce \
   --ckpt_path ${SAVE_DIR}/pg19/Llama-3.1-8B/lm_full_ce \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --num_training_samples 10000 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --pretrain_mode \
   --input_key "text" \
   --packing_samples \
   --bf16 \
   --num_processors 16 \
   --learning_rate 2e-5 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_wandb 'f81f2a236e712350a0ec153e02f43d1366c856a5' \
   --wandb_project 'baseline' \
   --wandb_run_name 'Llama-3.1-8B-lm_full_ce' \
   --ring_attn_size 4 \
   --ring_head_stride 2;