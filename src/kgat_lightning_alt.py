"""
Alternative KGAT model with train loss monitoring for LR scheduler
"""
from .kgat_lightning import KGATLightning
import torch


class KGATLightningAlt(KGATLightning):
    """KGAT model that monitors train loss instead of validation metrics"""
    
    def configure_optimizers(self):
        # Get base optimizer config
        config = super().configure_optimizers()
        
        # Change monitoring to train_loss which is always available
        if 'lr_scheduler' in config and isinstance(config['lr_scheduler'], dict):
            config['lr_scheduler']['monitor'] = 'train_loss'
            config['lr_scheduler']['mode'] = 'min'  # minimize loss
            
        return config