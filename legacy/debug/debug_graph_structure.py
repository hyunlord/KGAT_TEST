"""
Debug graph structure to understand why test items have poor embeddings
"""
import torch
import numpy as np
from data_module import KGATDataModule
from omegaconf import OmegaConf


def debug_graph_structure():
    """Check the graph structure and item connectivity"""
    config = OmegaConf.create({
        'data_dir': 'data/amazon-book',
        'batch_size': 32,
        'num_workers': 0,
        'neg_sample_size': 1
    })
    
    data_module = KGATDataModule(config)
    data_module.setup()
    
    print("\n=== Graph Structure Analysis ===")
    print(f"User-Item edge index shape: {data_module.edge_index_ui.shape}")
    print(f"KG edge index shape: {data_module.edge_index_kg.shape if data_module.edge_index_kg is not None else 'None'}")
    
    # Analyze which items are connected in the user-item graph
    edge_index_ui = data_module.edge_index_ui
    
    # Extract item nodes from the edge index (items are offset by n_users)
    item_nodes_in_graph = set()
    user_nodes_in_graph = set()
    
    for i in range(edge_index_ui.shape[1]):
        src, dst = edge_index_ui[0, i].item(), edge_index_ui[1, i].item()
        
        # Check if source is user or item
        if src < data_module.n_users:
            user_nodes_in_graph.add(src)
            item_nodes_in_graph.add(dst - data_module.n_users)
        else:
            item_nodes_in_graph.add(src - data_module.n_users)
            user_nodes_in_graph.add(dst)
    
    print(f"\nNodes in user-item graph:")
    print(f"Users: {len(user_nodes_in_graph)} out of {data_module.n_users}")
    print(f"Items: {len(item_nodes_in_graph)} out of {data_module.n_items}")
    
    # Check which items are NOT in the graph
    all_items = set(range(data_module.n_items))
    items_not_in_graph = all_items - item_nodes_in_graph
    
    print(f"\nItems NOT in user-item graph: {len(items_not_in_graph)}")
    if len(items_not_in_graph) > 0:
        print(f"Sample items not in graph: {sorted(list(items_not_in_graph))[:20]}...")
    
    # Check test items connectivity
    test_items_connected = 0
    test_items_total = 0
    
    for user_id, test_items in data_module.test_user_dict.items():
        for item in test_items:
            test_items_total += 1
            if item in item_nodes_in_graph:
                test_items_connected += 1
    
    print(f"\nTest items connectivity:")
    print(f"Total test items: {test_items_total}")
    print(f"Test items in graph: {test_items_connected}")
    print(f"Test items NOT in graph: {test_items_total - test_items_connected}")
    
    # Check if KG helps with connectivity
    if data_module.edge_index_kg is not None:
        items_in_kg = set()
        for i in range(data_module.edge_index_kg.shape[1]):
            src, dst = data_module.edge_index_kg[0, i].item(), data_module.edge_index_kg[1, i].item()
            if src < data_module.n_items:
                items_in_kg.add(src)
            if dst < data_module.n_items:
                items_in_kg.add(dst)
        
        print(f"\nKnowledge Graph connectivity:")
        print(f"Items in KG: {len(items_in_kg)}")
        print(f"Items in both UI graph and KG: {len(item_nodes_in_graph & items_in_kg)}")
        
        # Check if test items are in KG
        test_items_in_kg = 0
        for user_id, test_items in data_module.test_user_dict.items():
            for item in test_items:
                if item in items_in_kg:
                    test_items_in_kg += 1
        
        print(f"Test items in KG: {test_items_in_kg}")
    
    # Analyze the actual problem
    print("\n=== Problem Analysis ===")
    print("The issue is that test items (500+) are never seen during training.")
    print("They are not connected to any users in the training graph.")
    print("Their embeddings remain at random initialization, leading to poor predictions.")
    
    # Check if this is the intended behavior
    print("\n=== Data Design Check ===")
    train_item_range = set()
    test_item_range = set()
    
    for items in data_module.train_user_dict.values():
        train_item_range.update(items)
    
    for items in data_module.test_user_dict.values():
        test_item_range.update(items)
    
    overlap = train_item_range & test_item_range
    
    print(f"Train item range: [{min(train_item_range)}, {max(train_item_range)}]")
    print(f"Test item range: [{min(test_item_range)}, {max(test_item_range)}]")
    print(f"Overlapping items: {len(overlap)}")
    
    if len(overlap) == 0:
        print("\nWARNING: Train and test sets have COMPLETELY DISJOINT item sets!")
        print("This is likely the cause of zero metrics.")
        print("In recommendation systems, test items should be items that appear in training")
        print("but with different user-item interactions held out for testing.")


if __name__ == "__main__":
    debug_graph_structure()