import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import argparse

from kgat_lightning import KGATLightning
from kgat_improved import KGATImproved
from kgat_lightning_fixed import KGATLightningFixed
from data_module import KGATDataModule


class MetricsCallback(pl.Callback):
    """Custom callback for tracking and displaying metrics"""
    def __init__(self):
        self.best_metrics = {}
        
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        
        # Track best metrics
        for key, value in metrics.items():
            if 'val_' in key:
                metric_name = key.replace('val_', '')
                if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
        
        # Print current epoch metrics
        if trainer.current_epoch % 5 == 0:
            print(f"\n[Epoch {trainer.current_epoch}] Validation Metrics:")
            for key, value in metrics.items():
                if 'val_' in key:
                    print(f"  {key}: {value:.4f}")
    
    def on_fit_end(self, trainer, pl_module):
        print("\n=== Best Metrics ===")
        for key, value in self.best_metrics.items():
            print(f"{key}: {value:.4f}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    """Main training function with support for both original and improved models"""
    # Disable struct mode to allow dynamic key addition
    OmegaConf.set_struct(cfg, False)
    
    # Check which model to use
    model_type = cfg.get('model', {}).get('type', 'original')
    use_improved = cfg.get('use_improved_model', False)
    
    if model_type == 'kgat_fixed':
        model_name = 'Fixed'
    elif use_improved:
        model_name = 'Improved'
    else:
        model_name = 'Original'
    
    print(f"\n{'='*50}")
    print(f"Using {model_name} KGAT Model")
    print(f"Aggregator: {cfg.model.aggregator}")
    print(f"{'='*50}\n")
    
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(cfg.training.seed)
    
    # Convert to absolute path since Hydra changes working directory
    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.data.data_dir)
    
    # Create data module
    data_module = KGATDataModule(cfg.data)
    
    # Setup data module to get dataset statistics
    data_module.setup()
    
    # Update model config with dataset statistics
    cfg.model.n_users = data_module.n_users
    cfg.model.n_entities = data_module.n_entities
    cfg.model.n_relations = data_module.n_relations
    
    # Get dataset statistics
    stats = data_module.get_statistics()
    
    print(f"\nDataset Statistics:")
    print(f"  Users: {cfg.model.n_users}")
    print(f"  Entities: {cfg.model.n_entities}")
    print(f"  Relations: {cfg.model.n_relations}")
    print(f"  Train Interactions: {stats['n_train_interactions']}")
    print(f"  Test Interactions: {stats['n_test_interactions']}")
    print(f"  KG Triplets: {stats['n_kg_triples']}")
    
    # Create model
    if model_type == 'kgat_fixed':
        model = KGATLightningFixed(cfg.model)
    elif use_improved:
        model = KGATImproved(cfg.model)
    else:
        model = KGATLightning(cfg.model)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.model_save_dir,
        filename=f"{cfg.experiment.name}_" + "{epoch:02d}_{val_recall@20:.4f}",
        monitor='val_recall@20',
        mode='max',
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor='val_recall@20',
        patience=cfg.training.early_stopping_patience,
        mode='max',
        verbose=True
    )
    
    progress_bar = RichProgressBar()
    metrics_callback = MetricsCallback()
    
    # Create logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name=cfg.experiment.name,
        version=timestamp
    )
    
    # Create trainer with strategy handling
    trainer_kwargs = {
        'max_epochs': cfg.training.max_epochs,
        'accelerator': cfg.training.accelerator,
        'devices': cfg.training.devices,
        'precision': cfg.training.precision,
        'gradient_clip_val': cfg.training.gradient_clip_val,
        'accumulate_grad_batches': cfg.training.accumulate_grad_batches,
        'check_val_every_n_epoch': cfg.training.check_val_every_n_epoch,
        'log_every_n_steps': cfg.training.log_every_n_steps,
        'callbacks': [checkpoint_callback, early_stopping, progress_bar, metrics_callback],
        'logger': logger,
        'deterministic': True
    }
    
    # Handle strategy - use find_unused_parameters for DDP
    strategy = cfg.training.get('strategy', 'auto')
    if strategy == 'ddp':
        from pytorch_lightning.strategies import DDPStrategy
        trainer_kwargs['strategy'] = DDPStrategy(find_unused_parameters=True)
    else:
        trainer_kwargs['strategy'] = strategy
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Start training
    print("\nStarting training...")
    trainer.fit(model, data_module)
    
    # Test on best model
    print("\nTesting on best model...")
    trainer.test(model, data_module, ckpt_path='best')
    
    # Save final metrics
    final_metrics = {
        'best_val_recall@20': checkpoint_callback.best_model_score.item(),
        'best_epoch': checkpoint_callback.best_model_path.split('epoch=')[1].split('_')[0],
        'total_epochs': trainer.current_epoch
    }
    
    print("\n=== Training Complete ===")
    print(f"Best validation Recall@20: {final_metrics['best_val_recall@20']:.4f}")
    print(f"Best epoch: {final_metrics['best_epoch']}")
    print(f"Total epochs: {final_metrics['total_epochs']}")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    return final_metrics


if __name__ == "__main__":
    train()