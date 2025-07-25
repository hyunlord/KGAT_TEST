# KGAT PyTorch Lightning Implementation

Knowledge Graph Attention Network (KGAT) implementation using PyTorch Lightning for recommendation systems. This project includes training pipeline and evaluation methods comparing standard vs relation-enhanced recommendations.

## Features

- ✅ PyTorch Lightning based implementation
- ✅ Modular architecture with DataModule
- ✅ Hydra configuration management
- ✅ TensorBoard logging
- ✅ Model checkpointing and early stopping
- ✅ Two evaluation methods comparison:
  - Standard: User-Item similarity only
  - Enhanced: User+Relation-Item similarity

## Installation

```bash
# Basic installation
pip install -r requirements.txt

# For DeepSpeed support (optional)
pip install deepspeed

# For GPU monitoring (optional)
pip install gputil
```

## Quick Start

### 1. Prepare Data

```bash
# Create sample data for testing
python scripts/download_data.py --sample-only

# Or download specific dataset (instructions will be provided)
python scripts/download_data.py --dataset amazon-book
```

### 2. Train KGAT Model

#### Single GPU Training
```bash
# Train with default configuration
python src/train.py

# Train with custom configuration
python src/train.py data.batch_size=512 model.embed_dim=128

# Train with small configuration (for testing)
python src/train.py --config-name config_small
```

#### Multi-GPU Training
```bash
# Use all available GPUs with DDP (Recommended)
python src/train.py training.devices=-1 training.strategy=ddp

# Use specific number of GPUs
python src/train.py training.devices=4 training.strategy=ddp data.batch_size=4096

# Use multi-GPU configuration file
python src/train.py --config-name config_multi_gpu

# Use DeepSpeed for better memory efficiency
python src/train.py training.devices=-1 training.strategy=deepspeed_stage_2 training.precision=16
```

#### Full Training Example
```bash
# Download dataset first
python scripts/download_data.py --dataset amazon-book

# Train on Amazon-Book dataset with 4 GPUs
python src/train.py \
    data.data_dir=data/amazon-book \
    data.batch_size=4096 \
    training.devices=4 \
    training.strategy=ddp \
    training.precision=16 \
    training.max_epochs=200
```

### 3. Evaluate and Compare Methods

```bash
# Compare standard vs enhanced methods
python src/evaluate_comparison.py \
    --checkpoint logs/kgat_amazon-book/version_0/checkpoints/best.ckpt \
    --n-sample-users 20
```

## Project Structure

```
KGAT_TEST/
├── src/
│   ├── kgat_lightning.py      # PyTorch Lightning KGAT model
│   ├── data_module.py         # Data loading and preprocessing
│   ├── train.py               # Training script
│   ├── evaluator.py           # Evaluation methods
│   ├── compare_methods.py     # Method comparison utilities
│   └── evaluate_comparison.py # Compare trained model methods
├── configs/
│   ├── config.yaml            # Main configuration
│   └── config_small.yaml      # Small config for testing
├── scripts/
│   └── download_data.py       # Data download script
├── data/                      # Dataset directory
├── logs/                      # TensorBoard logs
├── models/                    # Saved models
└── results/                   # Evaluation results
```

## Configuration

Main configuration options in `configs/config.yaml`:

```yaml
data:
  data_dir: data/amazon-book
  batch_size: 1024
  
model:
  embed_dim: 64
  layer_dims: [32, 16]
  aggregator: bi-interaction
  
training:
  max_epochs: 1000
  early_stopping_patience: 20
  lr: 0.001
```

## Data Format

Required files in data directory:
- `train.txt`: User-item interactions (format: `user_id item_id1 item_id2 ...`)
- `test.txt`: Test interactions (same format)
- `kg_final.txt`: Knowledge graph triples (format: `head_entity relation_id tail_entity`)

## Training Monitoring

### TensorBoard Setup
```bash
# Install TensorBoard (already in requirements.txt)
pip install tensorboard

# Start TensorBoard
tensorboard --logdir logs/

# For remote server access
tensorboard --logdir logs/ --bind_all

# View at http://localhost:6006
```

See [TensorBoard Guide](docs/TensorBoard_Guide.md) for detailed setup and usage.

## Training Strategy Comparison

### DDP vs DeepSpeed
```bash
# Compare different distributed training strategies
python scripts/compare_strategies.py \
    --data-dir data/amazon-book \
    --devices 4 \
    --batch-size 2048 \
    --max-epochs 10
```

See [DDP vs DeepSpeed Guide](docs/DDP_vs_DeepSpeed.md) for detailed comparison.

## Results

The comparison script generates:
1. **Metrics comparison** (Recall@K, Precision@K, NDCG@K)
2. **Visualizations** (bar charts, improvement heatmaps)
3. **User-level analysis** (sample recommendations comparison)

Example output:
```
Standard Method (User-Item Similarity Only):
  Recall@20: 0.1234
  Precision@20: 0.0456
  NDCG@20: 0.0789

Enhanced Method (User+Relation-Item Similarity):
  Recall@20: 0.1456 (+18.0%)
  Precision@20: 0.0523 (+14.7%)
  NDCG@20: 0.0891 (+12.9%)
```

## Multi-GPU Training Guide

For detailed multi-GPU training instructions, see [Multi-GPU Guide](README_MULTI_GPU.md).

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{KGAT2019,
  author = {Wang, Xiang and He, Xiangnan and Cao, Yixin and Liu, Meng and Chua, Tat-Seng},
  title = {KGAT: Knowledge Graph Attention Network for Recommendation},
  booktitle = {KDD},
  year = {2019}
}
```