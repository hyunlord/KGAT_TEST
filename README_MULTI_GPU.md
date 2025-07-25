# Multi-GPU Training Guide for KGAT

This guide explains how to use multiple GPUs for training KGAT with PyTorch Lightning.

## Quick Start

### 1. Use All Available GPUs
```bash
# Using multi-GPU config file
python src/train.py --config-name config_multi_gpu

# Or modify devices parameter
python src/train.py training.devices=-1
```

### 2. Use Specific Number of GPUs
```bash
# Use 2 GPUs
python src/train.py training.devices=2

# Use specific GPU IDs
python src/train.py training.devices=[0,2,3]
```

### 3. Use the Convenience Script
```bash
./scripts/train_multi_gpu.sh
```

## Configuration Options

### Distributed Training Strategies

1. **DDP (Distributed Data Parallel)** - Recommended
   ```yaml
   training:
     strategy: ddp
     devices: -1  # All GPUs
   ```

2. **DDP Spawn** - For debugging
   ```yaml
   training:
     strategy: ddp_spawn
     devices: 4
   ```

3. **DeepSpeed** - For large models
   ```yaml
   training:
     strategy: deepspeed_stage_2
     precision: 16
   ```

### Batch Size Scaling

When using multiple GPUs, the effective batch size is:
```
effective_batch_size = batch_size * num_gpus * accumulate_grad_batches
```

Example configurations:
- 1 GPU: batch_size=1024
- 4 GPUs: batch_size=4096 (or keep 1024 and let DDP handle it)

### Learning Rate Scaling

The code automatically scales learning rate with the linear scaling rule:
```
effective_lr = base_lr * num_gpus
```

## Performance Tips

1. **Use Mixed Precision**
   ```yaml
   training:
     precision: 16  # Uses less memory, faster training
   ```

2. **Increase Batch Size**
   ```yaml
   data:
     batch_size: 4096  # For 4 GPUs
   ```

3. **More Workers**
   ```yaml
   data:
     num_workers: 16  # 4 per GPU
   ```

4. **Gradient Accumulation** (for limited memory)
   ```yaml
   training:
     accumulate_grad_batches: 4
   ```

## Monitoring

### Check GPU Usage
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Check which GPUs are being used
echo $CUDA_VISIBLE_DEVICES
```

### TensorBoard
```bash
tensorboard --logdir logs/
```

## Troubleshooting

### Out of Memory
- Reduce batch_size
- Use gradient accumulation
- Use precision=16
- Try deepspeed strategy

### Slow Data Loading
- Increase num_workers
- Use SSD for data storage
- Pre-process data

### Uneven GPU Usage
- Ensure batch_size is divisible by num_gpus
- Check data distribution

## Example Commands

### T4 GPU Server (Your Setup)
```bash
# Use all 4 T4 GPUs
python src/train.py \
    --config-name config_multi_gpu \
    data.dataset=amazon-book \
    data.batch_size=2048 \
    training.devices=4

# Use 2 GPUs with larger model
python src/train.py \
    training.devices=2 \
    model.embed_dim=128 \
    model.layer_dims=[64,32,16]
```

### Memory-Constrained Training
```bash
python src/train.py \
    training.devices=4 \
    training.strategy=deepspeed_stage_2 \
    training.precision=16 \
    training.accumulate_grad_batches=4 \
    data.batch_size=512
```

## Expected Speedup

With 4 T4 GPUs, you can expect:
- ~3.5x speedup over single GPU (due to communication overhead)
- Ability to use larger batch sizes
- Faster convergence with proper learning rate scaling

## Distributed Training Output

When training starts, you'll see:
```
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
Scaling learning rate from 0.001 to 0.004 for 4 GPUs
```