# TensorBoard Installation and Usage Guide

## Installation

### Method 1: With requirements.txt (Recommended)
```bash
# TensorBoard is already included in requirements.txt
pip install -r requirements.txt
```

### Method 2: Standalone Installation
```bash
# Install TensorBoard only
pip install tensorboard

# Or with specific version
pip install tensorboard>=2.7.0
```

### Method 3: With PyTorch
```bash
# If you already have PyTorch, it usually comes with TensorBoard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation
```bash
# Check if TensorBoard is installed
tensorboard --version

# Should output something like:
# TensorBoard 2.14.0
```

## Usage

### 1. Start TensorBoard Server
```bash
# Basic usage (from project root)
tensorboard --logdir logs/

# Specify port
tensorboard --logdir logs/ --port 6007

# Bind to all interfaces (for remote access)
tensorboard --logdir logs/ --bind_all

# With specific experiment
tensorboard --logdir logs/kgat_amazon-book/
```

### 2. Access TensorBoard

**Local Access:**
- Open browser: http://localhost:6006

**Remote Server Access:**
```bash
# Option 1: SSH Port Forwarding
# On your local machine:
ssh -L 6006:localhost:6006 username@server

# Then access: http://localhost:6006

# Option 2: Direct access (if --bind_all used)
# http://server-ip:6006
```

### 3. Multiple Experiments
```bash
# Compare multiple runs
tensorboard --logdir_spec=ddp:logs/ddp_run,deepspeed:logs/deepspeed_run

# Or use the logs directory with all experiments
tensorboard --logdir logs/ --reload_interval 30
```

## TensorBoard Features for KGAT

### 1. Scalars
Monitor training metrics:
- `train_loss`: Training loss per step
- `val_recall@K`: Validation recall at different K values
- `val_precision@K`: Validation precision
- `val_ndcg@K`: Validation NDCG
- `learning_rate`: Learning rate changes

### 2. Graphs
View model architecture:
- KGAT layer structure
- Attention mechanisms
- Embedding dimensions

### 3. Histograms
Analyze weight distributions:
- User embeddings
- Item embeddings
- Relation embeddings
- Attention weights

### 4. Time Series
Track training progress:
- Loss curves
- Metric improvements
- GPU utilization

## Advanced Usage

### 1. Custom Logging in Code
```python
# Already implemented in train.py
logger = TensorBoardLogger(
    save_dir="logs",
    name="kgat_experiment",
    version="version_1"
)

# Log custom metrics
self.log('custom_metric', value, prog_bar=True)
```

### 2. Comparing Runs
```bash
# Compare DDP vs DeepSpeed
tensorboard --logdir logs/ --tag_filter="strategy"

# Filter by dataset
tensorboard --logdir logs/ --tag_filter="amazon-book"
```

### 3. Export Data
```bash
# Export scalar data to CSV
tensorboard --logdir logs/ --export_scalars_to_csv=/path/to/output.csv
```

### 4. Profiling
```python
# Enable profiling in training
trainer = pl.Trainer(
    profiler="pytorch",
    logger=logger
)
```

## Troubleshooting

### Port Already in Use
```bash
# Kill existing TensorBoard process
pkill -f tensorboard

# Or use different port
tensorboard --logdir logs/ --port 6008
```

### Can't Access Remotely
```bash
# Check firewall
sudo ufw allow 6006

# Use SSH tunnel instead
ssh -L 6006:localhost:6006 user@server
```

### Logs Not Updating
```bash
# Force reload
tensorboard --logdir logs/ --reload_interval 5

# Clear cache
rm -rf ~/.tensorboard/
```

### Memory Issues
```bash
# Limit samples
tensorboard --logdir logs/ --samples_per_plugin scalars=1000
```

## Tips for Your Setup

### On T4 GPU Server
```bash
# Start TensorBoard with auto-reload
tensorboard --logdir logs/ --bind_all --reload_interval 30

# Access from local machine
ssh -L 6006:localhost:6006 hanadmin@search-dmdl-gpu6
# Then open: http://localhost:6006
```

### Monitor Multi-GPU Training
```bash
# Watch all GPU experiments
tensorboard --logdir logs/ --reload_multifile=true

# Filter by GPU count
tensorboard --logdir logs/ --tag_filter="devices:4"
```

### Best Practices
1. **Organize logs by experiment:**
   ```
   logs/
   ├── ddp_4gpu_batch4096/
   ├── deepspeed_stage2_4gpu/
   └── single_gpu_baseline/
   ```

2. **Use descriptive names:**
   ```python
   logger = TensorBoardLogger(
       save_dir="logs",
       name=f"kgat_{strategy}_{n_gpus}gpu_batch{batch_size}"
   )
   ```

3. **Log hyperparameters:**
   - Automatically logged by PyTorch Lightning
   - Check "HPARAMS" tab in TensorBoard

4. **Regular checkpoints:**
   - TensorBoard updates in real-time
   - No need to restart for new data