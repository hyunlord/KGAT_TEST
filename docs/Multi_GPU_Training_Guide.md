# Multi-GPU Training Guide for Original KGAT

This guide explains how to use the multi-GPU training script for the original KGAT implementation.

## Overview

The multi-GPU training script uses PyTorch's DistributedDataParallel (DDP) for efficient parallel training across multiple GPUs. DDP is preferred over DataParallel because it:

- Has better performance and scaling
- Uses separate processes for each GPU
- Reduces GPU memory usage
- Properly handles sparse tensor operations across GPUs

## Prerequisites

- PyTorch with CUDA support
- Multiple GPUs available on the system
- NCCL backend support (usually comes with PyTorch)

## Usage

### Basic Usage

To train with multiple GPUs, use the provided shell script:

```bash
# Train with 2 GPUs
./scripts/run_original_kgat_multi_gpu.sh --num_gpus 2

# Train with 4 GPUs
./scripts/run_original_kgat_multi_gpu.sh --num_gpus 4
```

### Command Line Options

The script supports the following options:

- `--num_gpus <N>`: Number of GPUs to use (default: 1)
- `--batch_size_per_gpu <N>`: Batch size per GPU (default: 1024)
- `--dataset <name>`: Dataset name: amazon-book, last-fm, yelp2018 (default: amazon-book)
- `--epochs <N>`: Number of training epochs (default: 1000)
- `--master_port <N>`: Master port for DDP communication (default: 29500)

### Examples

1. **Train with 2 GPUs on Amazon-Book dataset:**
   ```bash
   ./scripts/run_original_kgat_multi_gpu.sh --num_gpus 2 --dataset amazon-book
   ```

2. **Train with 4 GPUs and larger batch size:**
   ```bash
   ./scripts/run_original_kgat_multi_gpu.sh --num_gpus 4 --batch_size_per_gpu 2048
   ```

3. **Train on Last-FM dataset with 2 GPUs:**
   ```bash
   ./scripts/run_original_kgat_multi_gpu.sh --num_gpus 2 --dataset last-fm
   ```

## Batch Size Scaling

When using multiple GPUs, the total batch size is automatically scaled:

- Total CF batch size = `batch_size_per_gpu * num_gpus`
- Total KG batch size = `batch_size_per_gpu * 2 * num_gpus`

For example, with 4 GPUs and `batch_size_per_gpu=1024`:
- Total CF batch size = 4096
- Total KG batch size = 8192

## Implementation Details

### Sparse Tensor Handling

The implementation properly handles sparse tensors across GPUs by:

1. Creating sparse adjacency matrices on CPU
2. Moving them to the appropriate GPU when needed
3. Using `torch.sparse.mm` for sparse matrix multiplication
4. Ensuring sparse tensors are on the correct device before operations

### Data Distribution

The training data is distributed across GPUs as follows:

1. Each GPU gets a portion of the batch
2. User lists are shuffled with different seeds per GPU
3. Loss is aggregated across all GPUs using `all_reduce`
4. Only rank 0 performs evaluation and logging

### Synchronization

The implementation includes proper synchronization:

- `dist.barrier()` at the end of each epoch
- Loss aggregation using `dist.all_reduce`
- Model saving only on rank 0
- Evaluation only on rank 0

## Performance Tips

1. **Batch Size**: Increase `batch_size_per_gpu` for better GPU utilization
2. **GPU Count**: Performance scales well up to 4-8 GPUs
3. **Network**: Ensure fast inter-GPU communication (NVLink preferred)
4. **Memory**: Monitor GPU memory usage and adjust batch size if needed

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size_per_gpu`
   - Check if other processes are using GPU memory

2. **Process Group Initialization Failed**
   - Check if the master port is already in use
   - Try a different port with `--master_port`

3. **Uneven GPU Utilization**
   - This is normal for the last batch of each epoch
   - Consider using a batch size that divides evenly

### Debugging

To debug multi-GPU training:

1. Check GPU availability:
   ```bash
   nvidia-smi
   ```

2. Monitor GPU usage during training:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. Check logs in `logs/<dataset>/train_multi_gpu.log`

## Comparison with Single GPU

The multi-GPU implementation maintains the same model architecture and training logic as the single GPU version, with these key differences:

1. Data is distributed across GPUs
2. Model is wrapped with DistributedDataParallel
3. Loss is aggregated across all processes
4. Each GPU maintains its own random seed for data shuffling

## Advanced Usage

### Custom Configurations

You can modify the Python script directly for more advanced configurations:

```python
# In src/train_original_multi_gpu.py
# Adjust gradient accumulation, learning rate scaling, etc.
```

### Mixed Precision Training

To enable mixed precision training (not implemented by default):

1. Add `--fp16` flag to the argument parser
2. Use `torch.cuda.amp.autocast()` and `GradScaler`
3. This can significantly speed up training on modern GPUs

## References

- [PyTorch Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch DDP Notes](https://pytorch.org/docs/stable/notes/ddp.html)
- [Original KGAT Paper](https://arxiv.org/abs/1905.07854)