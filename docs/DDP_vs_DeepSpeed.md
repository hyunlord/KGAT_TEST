# DDP vs DeepSpeed Comparison Guide

## Overview

This document compares two distributed training strategies for KGAT:
- **DDP (Distributed Data Parallel)**: PyTorch's native distributed training
- **DeepSpeed**: Microsoft's optimization library for large-scale training

## Quick Comparison

| Feature | DDP | DeepSpeed |
|---------|-----|-----------|
| Setup Complexity | Simple | Moderate |
| Memory Efficiency | Good | Excellent |
| Speed | Fast | Faster (10-20%) |
| Large Model Support | Limited | Excellent |
| Mixed Precision | Yes | Yes (optimized) |
| CPU Offloading | No | Yes |

## Training Commands

### DDP Training
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=ddp \
    training.precision=16 \
    data.batch_size=4096
```

### DeepSpeed Stage 1 (Optimizer State Partitioning)
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=deepspeed_stage_1 \
    training.precision=16 \
    data.batch_size=4096
```

### DeepSpeed Stage 2 (Optimizer + Gradient Partitioning)
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=deepspeed_stage_2 \
    training.precision=16 \
    data.batch_size=4096
```

### DeepSpeed Stage 3 (Full Partitioning + CPU Offload)
```bash
python src/train.py \
    data.data_dir=data/amazon-book \
    training.devices=4 \
    training.strategy=deepspeed_stage_3 \
    training.precision=16 \
    data.batch_size=2048  # Smaller batch size
```

## Performance Comparison Script

Run automated comparison:
```bash
# Compare DDP vs DeepSpeed stages
python scripts/compare_strategies.py \
    --data-dir data/amazon-book \
    --devices 4 \
    --batch-size 2048 \
    --max-epochs 10 \
    --strategies ddp deepspeed_stage_1 deepspeed_stage_2

# Include Stage 3 (requires more memory)
python scripts/compare_strategies.py \
    --strategies ddp deepspeed_stage_1 deepspeed_stage_2 deepspeed_stage_3 \
    --test-stage-3
```

## Expected Results

### Memory Usage (4x Tesla T4 GPUs)
- **DDP**: ~12GB per GPU
- **DeepSpeed Stage 1**: ~10GB per GPU (20% reduction)
- **DeepSpeed Stage 2**: ~8GB per GPU (35% reduction)
- **DeepSpeed Stage 3**: ~6GB per GPU (50% reduction)

### Training Speed
- **DDP**: Baseline (1.0x)
- **DeepSpeed Stage 1**: 1.1x faster
- **DeepSpeed Stage 2**: 1.15x faster
- **DeepSpeed Stage 3**: 0.9x (slower due to CPU offload)

### When to Use What?

#### Use DDP when:
- Model fits comfortably in GPU memory
- You want simplest setup
- Debugging distributed training issues

#### Use DeepSpeed Stage 1 when:
- Model barely fits in memory
- You want 10-20% memory savings
- Minimal code changes needed

#### Use DeepSpeed Stage 2 when:
- Need significant memory savings
- Training very large models
- Want best speed/memory trade-off

#### Use DeepSpeed Stage 3 when:
- Model doesn't fit even with Stage 2
- Have fast CPU-GPU interconnect
- Training extremely large models

## Troubleshooting

### DeepSpeed Installation
```bash
# For CUDA 11.x
pip install deepspeed

# With specific CUDA version
DS_BUILD_CUDA_EXT=1 pip install deepspeed
```

### Common Issues

1. **NCCL Errors with DeepSpeed**
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_P2P_DISABLE=1  # If P2P issues
   ```

2. **Memory Fragmentation**
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

3. **DeepSpeed Config Tuning**
   ```python
   # Custom DeepSpeed config
   training.strategy = {
       "type": "deepspeed",
       "config": {
           "stage": 2,
           "offload_optimizer": True,
           "offload_parameters": False
       }
   }
   ```

## Monitoring

### Real-time GPU Usage
```bash
# Watch GPU memory and utilization
watch -n 1 nvidia-smi

# Detailed process view
nvidia-smi pmon -i 0,1,2,3
```

### Training Metrics
```bash
# TensorBoard
tensorboard --logdir logs/

# Check at http://localhost:6006
```

## Recommendations for Your Setup (4x T4 GPUs)

1. **For Standard KGAT Training**: Use DDP
   - Simple, reliable, fast enough
   - Good for models up to 500M parameters

2. **For Large Embeddings**: Use DeepSpeed Stage 1
   - When embedding tables are very large
   - Saves memory with minimal overhead

3. **For Memory-Constrained Scenarios**: Use DeepSpeed Stage 2
   - Best balance of speed and memory
   - Allows larger batch sizes

4. **For Experimental/Research**: Try all strategies
   - Use comparison script to find optimal settings
   - Document results for reproducibility