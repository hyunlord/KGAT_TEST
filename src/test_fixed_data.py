"""
Test KGAT with properly formatted data
"""
import torch
import numpy as np
from data_module import KGATDataModule
from kgat_lightning import KGATLightning
from omegaconf import OmegaConf
import pytorch_lightning as pl
# from pytorch_lightning.callbacks import RichProgressBar


def test_with_fixed_data():
    """Test KGAT training with fixed data"""
    # Setup data
    config = OmegaConf.create({
        'data_dir': 'data/amazon-book-fixed',
        'batch_size': 128,
        'num_workers': 0,
        'neg_sample_size': 1
    })
    
    data_module = KGATDataModule(config)
    data_module.setup()
    
    # Print data statistics
    stats = data_module.get_statistics()
    print("\n=== Data Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Setup model
    model_config = OmegaConf.create({
        'n_users': data_module.n_users,
        'n_entities': data_module.n_entities,
        'n_relations': data_module.n_relations,
        'embed_dim': 64,
        'layer_dims': [32, 16],
        'dropout': 0.1,
        'aggregator': 'bi-interaction',
        'reg_weight': 1e-5,
        'lr': 0.001
    })
    
    model = KGATLightning(model_config)
    
    # Quick training test
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        # callbacks=[RichProgressBar()],
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=0  # Prevent reloading data
    )
    
    print("\n=== Starting Training ===")
    trainer.fit(model, data_module)
    
    # Test evaluation
    print("\n=== Testing Evaluation ===")
    test_results = trainer.test(model, data_module)
    
    print("\n=== Test Results ===")
    for key, value in test_results[0].items():
        print(f"{key}: {value:.4f}")
    
    # Manual evaluation check
    print("\n=== Manual Evaluation Check ===")
    model.eval()
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    
    with torch.no_grad():
        # Get embeddings
        user_embeds, entity_embeds = model(
            batch['edge_index_ui'],
            batch.get('edge_index_kg', None),
            batch.get('edge_type_kg', None)
        )
        
        # Check one user
        user_id = batch['eval_user_ids'][0]
        user_embed = user_embeds[user_id]
        scores = torch.matmul(user_embed, entity_embeds.t())
        
        # Mask training items
        train_items = batch['train_items'][0]
        scores[train_items] = -float('inf')
        
        # Get top 20
        top_scores, top_indices = torch.topk(scores, 20)
        
        # Check against test items
        test_items = set(batch['test_items'][0])
        recommended = set(top_indices.cpu().numpy())
        hits = recommended & test_items
        
        print(f"\nUser {user_id.item()}:")
        print(f"Test items: {test_items}")
        print(f"Top 20 recommendations: {recommended}")
        print(f"Hits: {hits}")
        print(f"Recall@20: {len(hits) / len(test_items) if test_items else 0:.4f}")


if __name__ == "__main__":
    test_with_fixed_data()