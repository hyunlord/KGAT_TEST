import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

from kgat_lightning import KGATLightning
from data_module import KGATDataModule


class MetricsCallback(pl.Callback):
    """Custom callback to track and print metrics"""
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


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    """Main training function"""
    print(OmegaConf.to_yaml(cfg))
    
    # Set random seed
    pl.seed_everything(cfg.training.seed)
    
    # Initialize data module
    data_module = KGATDataModule(cfg.data)
    data_module.setup()
    
    # Get data statistics for model initialization
    stats = data_module.get_statistics()
    cfg.model.n_users = stats['n_users']
    cfg.model.n_entities = stats['n_entities']
    cfg.model.n_relations = stats['n_relations']
    
    print(f"\nData Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Initialize model
    model = KGATLightning(cfg.model)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_recall@20',
            mode='max',
            save_top_k=3,
            filename='kgat-{epoch:02d}-{val_recall@20:.3f}',
            save_last=True,
            verbose=True
        ),
        EarlyStopping(
            monitor='val_recall@20',
            mode='max',
            patience=cfg.training.early_stopping_patience,
            verbose=True
        ),
        RichProgressBar(),
        MetricsCallback()
    ]
    
    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name=f"kgat_{cfg.data.dataset}",
        version=timestamp
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.training.check_val_every_n_epoch,
        log_every_n_steps=cfg.training.log_every_n_steps,
        precision=cfg.training.precision,
        deterministic=True
    )
    
    # Train model
    print("\nStarting training...")
    trainer.fit(model, data_module)
    
    # Test model
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"\nBest validation metrics:")
    for key, value in callbacks[3].best_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTest results:")
    for key, value in test_results[0].items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nModel checkpoints saved in: {trainer.checkpoint_callback.dirpath}")
    print(f"Tensorboard logs saved in: {logger.log_dir}")
    
    # Save final model
    final_model_path = os.path.join(cfg.training.model_save_dir, f"kgat_final_{timestamp}.pth")
    os.makedirs(cfg.training.model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    return test_results[0]


if __name__ == "__main__":
    train()