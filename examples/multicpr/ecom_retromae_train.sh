WANDB_MODE=disabled \
NCCL_P2P_DISABLE="1" \
NCCL_IB_DISABLE="1" \
CUDA_VISIBLE_DEVICES="5" \
/data/miniconda3/envs/deepnlp/bin/torchrun --nproc_per_node 1 -m EmbedBoost.retromae.pretrain.run \
  --output_dir output/multicpr/ecom/v1/models/ernie-3.0-medium-zh-retromae \
  --overwrite_output_dir \
  --data_dir output/multicpr/ecom/v1/datasets/retromae \
  --do_train True \
  --model_name_or_path /data/pretrained_models/ernie-3.0-medium-zh \
  --per_device_train_batch_size 64 \
  --max_seq_length 256 \
  --pretrain_method retromae \
  --fp16 False \
  --dataloader_num_workers 4 \
  --weight_decay 0.01 \
  --encoder_mlm_probability 0.3 \
  --decoder_mlm_probability 0.5 \
  --num_train_epochs 2 \
  --warmup_ratio 0.1 \
  --learning_rate 6e-5 \
  --save_steps 20000
