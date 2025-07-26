"""
Debug data consistency between train and test
"""
from data_module import KGATDataModule
from omegaconf import OmegaConf


def check_data_consistency():
    """Check consistency of data dimensions across train/val/test"""
    config = OmegaConf.create({
        'data_dir': 'data/amazon-book-fixed',
        'batch_size': 128,
        'num_workers': 0,
        'neg_sample_size': 1
    })
    
    data_module = KGATDataModule(config)
    data_module.setup()
    
    print("=== Data Statistics ===")
    print(f"n_users: {data_module.n_users}")
    print(f"n_items: {data_module.n_items}")
    print(f"n_entities: {data_module.n_entities}")
    print(f"n_relations: {data_module.n_relations}")
    
    # Check edge indices
    print(f"\n=== Edge Index Shapes ===")
    print(f"UI edge index: {data_module.edge_index_ui.shape}")
    if data_module.edge_index_kg is not None:
        print(f"KG edge index: {data_module.edge_index_kg.shape}")
    
    # Check max indices in edges
    print(f"\n=== Max Indices in Edges ===")
    ui_max = data_module.edge_index_ui.max().item()
    ui_min = data_module.edge_index_ui.min().item()
    print(f"UI edge index - min: {ui_min}, max: {ui_max}")
    print(f"Expected max: {data_module.n_users + data_module.n_items - 1}")
    
    if ui_max >= data_module.n_users + data_module.n_items:
        print("ERROR: UI edge index contains out-of-bounds indices!")
    
    if data_module.edge_index_kg is not None:
        kg_max = data_module.edge_index_kg.max().item()
        kg_min = data_module.edge_index_kg.min().item()
        print(f"\nKG edge index - min: {kg_min}, max: {kg_max}")
        print(f"Expected max for items: {data_module.n_items - 1}")
        
        if kg_max >= data_module.n_items:
            print("ERROR: KG edge index contains out-of-bounds item indices!")
    
    # Check each dataloader
    print(f"\n=== Checking Dataloaders ===")
    
    # Train
    train_loader = data_module.train_dataloader()
    train_batch = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  Edge index UI shape: {train_batch['edge_index_ui'].shape}")
    print(f"  Edge index UI max: {train_batch['edge_index_ui'].max().item()}")
    
    # Val
    val_loader = data_module.val_dataloader()
    val_batch = next(iter(val_loader))
    print(f"\nVal batch:")
    print(f"  Edge index UI shape: {val_batch['edge_index_ui'].shape}")
    print(f"  Edge index UI max: {val_batch['edge_index_ui'].max().item()}")
    
    # Test
    test_loader = data_module.test_dataloader()
    test_batch = next(iter(test_loader))
    print(f"\nTest batch:")
    print(f"  Edge index UI shape: {test_batch['edge_index_ui'].shape}")
    print(f"  Edge index UI max: {test_batch['edge_index_ui'].max().item()}")
    
    # Check if edge indices are the same
    print(f"\n=== Edge Index Consistency ===")
    print(f"Train and Val edge indices equal: {(train_batch['edge_index_ui'] == val_batch['edge_index_ui']).all().item()}")
    print(f"Train and Test edge indices equal: {(train_batch['edge_index_ui'] == test_batch['edge_index_ui']).all().item()}")


if __name__ == "__main__":
    check_data_consistency()