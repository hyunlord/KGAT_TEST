"""
Debug evaluation metrics to find why they are all zeros
"""
import torch
import numpy as np
from data_module import KGATDataModule
from kgat_lightning import KGATLightning
from omegaconf import OmegaConf
from collections import defaultdict


def debug_evaluation_step():
    """Debug evaluation metrics calculation"""
    # Setup
    config = OmegaConf.create({
        'data_dir': 'data/amazon-book',
        'batch_size': 32,
        'num_workers': 0,
        'neg_sample_size': 1
    })
    
    data_module = KGATDataModule(config)
    data_module.setup()
    
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
    model.eval()
    
    # Get one batch from validation dataloader
    val_loader = data_module.val_dataloader()
    batch = next(iter(val_loader))
    
    print(f"\n=== Batch Information ===")
    print(f"Number of users in batch: {len(batch['eval_user_ids'])}")
    print(f"Edge index UI shape: {batch['edge_index_ui'].shape}")
    if batch['edge_index_kg'] is not None:
        print(f"Edge index KG shape: {batch['edge_index_kg'].shape}")
    
    # Run forward pass
    with torch.no_grad():
        user_embeds, entity_embeds = model(
            batch['edge_index_ui'],
            batch.get('edge_index_kg', None),
            batch.get('edge_type_kg', None)
        )
    
    print(f"\n=== Embedding Information ===")
    print(f"User embeddings shape: {user_embeds.shape}")
    print(f"Entity embeddings shape: {entity_embeds.shape}")
    
    # Debug metric calculation for first few users
    metrics = defaultdict(list)
    
    for idx in range(min(5, len(batch['eval_user_ids']))):
        user_id = batch['eval_user_ids'][idx]
        user_embed = user_embeds[user_id]
        
        # Calculate scores with all items
        scores = torch.matmul(user_embed, entity_embeds.t())
        
        # Get train and test items
        train_items = batch['train_items'][idx]
        test_items = batch['test_items'][idx]
        
        print(f"\n=== User {user_id.item()} ===")
        print(f"Train items: {train_items}")
        print(f"Test items: {test_items}")
        print(f"Scores shape: {scores.shape}")
        print(f"Scores stats - mean: {scores.mean():.4f}, std: {scores.std():.4f}, max: {scores.max():.4f}, min: {scores.min():.4f}")
        
        # Mask out training items
        scores_before_masking = scores.clone()
        scores[train_items] = -float('inf')
        
        # Check if masking worked
        print(f"Number of -inf scores after masking: {(scores == -float('inf')).sum().item()}")
        print(f"Expected number of masked items: {len(train_items)}")
        
        # Get top k items
        k = 20
        k_actual = min(k, scores.size(0))
        top_scores, top_indices = torch.topk(scores, k_actual)
        
        print(f"\nTop {k_actual} recommendations:")
        print(f"Indices: {top_indices.tolist()}")
        print(f"Scores: {top_scores.tolist()[:5]}... (showing first 5)")
        
        # Check if any test items are in top k
        recommended = set(top_indices.cpu().numpy())
        if torch.is_tensor(test_items):
            ground_truth = set(test_items.cpu().numpy())
        else:
            ground_truth = set(test_items)
            
        hits = recommended & ground_truth
        print(f"\nGround truth items: {ground_truth}")
        print(f"Recommended items: {recommended}")
        print(f"Hits: {hits}")
        
        # Calculate metrics
        recall = len(hits) / len(ground_truth) if ground_truth else 0
        precision = len(hits) / k_actual
        
        print(f"\nMetrics:")
        print(f"Recall@{k}: {recall:.4f}")
        print(f"Precision@{k}: {precision:.4f}")
        
        # Check specific test items
        if len(test_items) > 0:
            test_item = test_items[0] if isinstance(test_items, list) else test_items[0].item()
            test_item_score = scores_before_masking[test_item]
            test_item_rank = (scores_before_masking > test_item_score).sum().item() + 1
            print(f"\nFirst test item {test_item}:")
            print(f"  Score: {test_item_score:.4f}")
            print(f"  Rank: {test_item_rank}")
            
            # Check if test item is in the valid range
            if test_item >= entity_embeds.size(0):
                print(f"  WARNING: Test item {test_item} is out of range! Max item index: {entity_embeds.size(0) - 1}")


def check_data_overlap():
    """Check if there's overlap between train and test sets"""
    config = OmegaConf.create({
        'data_dir': 'data/amazon-book',
        'batch_size': 32,
        'num_workers': 0,
        'neg_sample_size': 1
    })
    
    data_module = KGATDataModule(config)
    data_module.setup()
    
    print("\n=== Data Overlap Check ===")
    
    overlap_users = 0
    total_overlap_items = 0
    
    for user_id in range(min(10, data_module.n_users)):
        train_items = set(data_module.train_user_dict.get(user_id, []))
        test_items = set(data_module.test_user_dict.get(user_id, []))
        
        overlap = train_items & test_items
        if overlap:
            overlap_users += 1
            total_overlap_items += len(overlap)
            print(f"User {user_id}: {len(overlap)} overlapping items: {overlap}")
    
    print(f"\nTotal users with overlap: {overlap_users}")
    print(f"Total overlapping items: {total_overlap_items}")
    
    # Check item ID ranges
    all_train_items = set()
    all_test_items = set()
    
    for items in data_module.train_user_dict.values():
        all_train_items.update(items)
    
    for items in data_module.test_user_dict.values():
        all_test_items.update(items)
    
    print(f"\nItem ID ranges:")
    print(f"Train items: min={min(all_train_items)}, max={max(all_train_items)}")
    print(f"Test items: min={min(all_test_items)}, max={max(all_test_items)}")
    print(f"Number of items in model: {data_module.n_items}")
    
    # Check if test items exceed n_items
    out_of_range_items = [item for item in all_test_items if item >= data_module.n_items]
    if out_of_range_items:
        print(f"\nWARNING: {len(out_of_range_items)} test items are out of range!")
        print(f"Out of range items: {sorted(out_of_range_items)[:10]}... (showing first 10)")


if __name__ == "__main__":
    print("=== Debugging Evaluation Metrics ===")
    
    # First check data overlap
    check_data_overlap()
    
    # Then debug evaluation
    debug_evaluation_step()