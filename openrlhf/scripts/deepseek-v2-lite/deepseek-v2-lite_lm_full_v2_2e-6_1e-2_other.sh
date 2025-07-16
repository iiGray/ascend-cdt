# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export HF_DATASETS_CACHE="/mnt/hwfile/opendatalab/tangzecheng/cache"

SAVE_DIR='/data/long/deepseek'
MODEL_PATH='/data/DeepSeek-V2-Lite'


# deepspeed --master_port 29506 --include localhost:0,1,2,3,4,5,6,7 cli/context_denoise/train_cdt.py \

deepspeed --module train_cdt \
   --max_len 32000 \
   --dataset '/data/datas/pg19' \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain ${MODEL_PATH} \
   --save_dir ${SAVE_DIR} \
   --save_path ${SAVE_DIR}/model/pg19/deepseek-v2-lite/other_gradient_large_pos_1e-5_1e-2_32k \
   --ckpt_path ${SAVE_DIR}/ckpt/pg19/deepseek-v2-lite/other_gradient_large_pos_1e-5_1e-2_32k \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --num_training_samples 10000 \
   --zero_stage 2 \
   --max_ckpt_num 20 \
   --max_epochs 2 \
   --pretrain_mode \
   --adv_epsilon 0.01 \
   --input_key "text" \
   --packing_samples \
   --bf16 \
   --num_processors 32 \
   --learning_rate 1e-5 \
   --flash_attn \
   --gradient_checkpointing \
   --disable_fast_tokenizer \
   --use_tensorboard '/data/long/tb_ds' \
   --wandb_run_name 'deepseek-v2-lite_other_gradient_large_pos_1e-5_1e-2_32k' \
   --perturb_type 'embedding' \
   --direction 'other' \
   --ring_attn_size 4 \
   --ring_head_stride 2;