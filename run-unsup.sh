#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=8

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THRE16DS=8

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path roberta-base \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-robert-base-uncased \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --mlp_only_train \
    --do_eval \
    --do_train \
    "$@"


# 