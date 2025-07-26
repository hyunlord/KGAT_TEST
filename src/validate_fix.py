"""
Validate that the data fix resolves the zero metrics issue
"""
import torch
import numpy as np
from data_module import KGATDataModule
from kgat_lightning import KGATLightning
from omegaconf import OmegaConf
from collections import defaultdict


def validate_metrics():
    """Test that metrics are non-zero with fixed data"""
    # Load both datasets
    datasets = {
        'broken': 'data/amazon-book',
        'fixed': 'data/amazon-book-fixed'
    }
    
    results = {}
    
    for name, data_dir in datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing with {name} dataset: {data_dir}")
        print('='*60)
        
        # Setup data
        config = OmegaConf.create({
            'data_dir': data_dir,
            'batch_size': 128,
            'num_workers': 0,
            'neg_sample_size': 1
        })
        
        data_module = KGATDataModule(config)
        data_module.setup()
        
        # Print data stats
        stats = data_module.get_statistics()
        print(f"\nData Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Check item ranges
        train_items = set()
        test_items = set()
        
        for items in data_module.train_user_dict.values():
            train_items.update(items)
        
        for items in data_module.test_user_dict.values():
            test_items.update(items)
        
        print(f"\nItem ranges:")
        print(f"  Train items: [{min(train_items) if train_items else 'N/A'}, {max(train_items) if train_items else 'N/A'}]")
        print(f"  Test items: [{min(test_items) if test_items else 'N/A'}, {max(test_items) if test_items else 'N/A'}]")
        print(f"  Test items in train: {len(test_items & train_items)} / {len(test_items)} ({100 * len(test_items & train_items) / len(test_items) if test_items else 0:.1f}%)")
        
        # Setup model
        model_config = OmegaConf.create({
            'n_users': data_module.n_users,
            'n_entities': data_module.n_entities,
            'n_relations': data_module.n_relations,
            'embed_dim': 64,
            'layer_dims': [32, 16],
            'dropout': 0.0,  # No dropout for testing
            'aggregator': 'bi-interaction',
            'reg_weight': 1e-5,
            'lr': 0.001
        })
        
        model = KGATLightning(model_config)
        model.eval()
        
        # Get embeddings
        with torch.no_grad():
            user_embeds, entity_embeds = model(
                data_module.edge_index_ui,
                data_module.edge_index_kg,
                data_module.edge_type_kg
            )
        
        print(f"\nEmbedding stats:")
        print(f"  User embeds: mean={user_embeds.mean():.4f}, std={user_embeds.std():.4f}")
        print(f"  Entity embeds: mean={entity_embeds.mean():.4f}, std={entity_embeds.std():.4f}")
        
        # Test metrics on a few users
        val_loader = data_module.val_dataloader()
        batch = next(iter(val_loader))
        
        metrics = defaultdict(list)
        n_test_users = min(10, len(batch['eval_user_ids']))
        
        for idx in range(n_test_users):
            user_id = batch['eval_user_ids'][idx]
            user_embed = user_embeds[user_id]
            
            # Calculate scores
            scores = torch.matmul(user_embed, entity_embeds.t())
            
            # Mask training items
            train_items_user = batch['train_items'][idx]
            scores[train_items_user] = -float('inf')
            
            # Get recommendations
            k = 20
            top_scores, top_indices = torch.topk(scores, k)
            recommended = set(top_indices.cpu().numpy())
            
            # Get ground truth
            test_items_user = batch['test_items'][idx]
            if torch.is_tensor(test_items_user):
                ground_truth = set(test_items_user.cpu().numpy())
            else:
                ground_truth = set(test_items_user)
            
            # Calculate metrics
            hits = recommended & ground_truth
            recall = len(hits) / len(ground_truth) if ground_truth else 0
            
            metrics['recall@20'].append(recall)
            
            # Debug first user
            if idx == 0:
                print(f"\nExample user {user_id.item()}:")
                print(f"  Train items: {len(train_items_user)} items")
                print(f"  Test items: {ground_truth}")
                print(f"  Top recommendations: {list(recommended)[:5]}...")
                print(f"  Hits: {hits}")
                print(f"  Recall@20: {recall:.4f}")
        
        # Average metrics
        avg_recall = np.mean(metrics['recall@20'])
        results[name] = avg_recall
        
        print(f"\nAverage Recall@20 for {n_test_users} users: {avg_recall:.4f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for name, recall in results.items():
        print(f"{name} dataset - Recall@20: {recall:.4f}")
    
    if results.get('fixed', 0) > results.get('broken', 0):
        print("\n✅ SUCCESS: Fixed dataset shows improved metrics!")
    else:
        print("\n❌ Issue persists: Metrics are still low")


if __name__ == "__main__":
    validate_metrics()