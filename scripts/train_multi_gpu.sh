#!/bin/bash
# Multi-GPU training script for KGAT

# Set CUDA devices (optional - will use all by default)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Training with multi-GPU configuration
echo "Starting multi-GPU training on all available GPUs..."
python src/train.py --config-name config_multi_gpu

# Alternative: Specify number of GPUs
# python src/train.py training.devices=2

# Alternative: Specify specific GPUs
# python src/train.py training.devices=[0,1,2]

# Alternative: Use DeepSpeed for even better performance
# python src/train.py training.strategy=deepspeed_stage_2 training.precision=16