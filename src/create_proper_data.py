"""
Create proper synthetic data for KGAT with correct train/test split
"""
import os
import random
import numpy as np
from collections import defaultdict


def create_proper_sample_data(data_dir, n_users=1000, n_items=2000, n_relations=5):
    """
    Create sample data with proper train/test split
    
    The key is that test items must be items that appear in training,
    but we hold out some user-item interactions for testing.
    """
    print(f"\nCreating proper sample data in {data_dir}")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate user-item interactions
    all_interactions = defaultdict(set)
    
    # Each user interacts with 10-50 items
    for user in range(n_users):
        n_interactions = random.randint(10, 50)
        # Use zipf distribution for more realistic item popularity
        items = np.random.zipf(1.5, n_interactions) % n_items
        all_interactions[user].update(items)
    
    # Split interactions into train/test (80/20 split)
    train_interactions = defaultdict(list)
    test_interactions = defaultdict(list)
    
    # First, ensure all items appear in at least one training interaction
    item_coverage = defaultdict(int)
    
    for user, items in all_interactions.items():
        items = list(items)
        if len(items) < 3:  # Skip users with too few interactions
            train_interactions[user] = items
            for item in items:
                item_coverage[item] += 1
            continue
            
        # Shuffle and split
        random.shuffle(items)
        n_test = max(1, min(len(items) // 5, len(items) - 2))  # 20% for test, but keep at least 2 for train
        
        train_items = items[:-n_test]
        test_items = items[-n_test:]
        
        # Ensure test items appear in training for at least one other user
        final_test_items = []
        for item in test_items:
            if item_coverage[item] > 0:  # Item already in some training set
                final_test_items.append(item)
            else:  # Move to train to ensure coverage
                train_items.append(item)
                item_coverage[item] += 1
        
        # Update coverage
        for item in train_items:
            item_coverage[item] += 1
        
        train_interactions[user] = train_items
        test_interactions[user] = final_test_items if final_test_items else []
    
    # Write train.txt
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        for user in range(n_users):
            if user in train_interactions and train_interactions[user]:
                items_str = ' '.join(map(str, train_interactions[user]))
                f.write(f"{user} {items_str}\n")
    
    # Write test.txt
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for user in range(n_users):
            if user in test_interactions and test_interactions[user]:
                items_str = ' '.join(map(str, test_interactions[user]))
                f.write(f"{user} {items_str}\n")
    
    # Create knowledge graph
    # Items can have relations with other items
    kg_triples = []
    
    # Popular items have more connections
    item_popularity = defaultdict(int)
    for items in all_interactions.values():
        for item in items:
            item_popularity[item] += 1
    
    # Sort items by popularity
    popular_items = sorted(item_popularity.keys(), key=lambda x: item_popularity[x], reverse=True)
    
    # Create KG triples
    for i, item in enumerate(popular_items[:n_items//2]):  # Top 50% of items
        # Number of relations proportional to popularity
        n_kg_relations = min(20, max(2, item_popularity[item] // 10))
        
        for _ in range(n_kg_relations):
            relation = random.randint(0, n_relations - 1)
            # Connect to other popular items
            target_idx = random.randint(0, min(len(popular_items) - 1, n_items // 4))
            target_item = popular_items[target_idx]
            
            if target_item != item:
                kg_triples.append((item, relation, target_item))
    
    # Write kg_final.txt
    with open(os.path.join(data_dir, 'kg_final.txt'), 'w') as f:
        for head, rel, tail in kg_triples:
            f.write(f"{head} {rel} {tail}\n")
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Relations: {n_relations}")
    
    n_train = sum(len(items) for items in train_interactions.values())
    n_test = sum(len(items) for items in test_interactions.values())
    
    print(f"Train interactions: {n_train}")
    print(f"Test interactions: {n_test}")
    print(f"KG triples: {len(kg_triples)}")
    
    # Verify data integrity
    train_items = set()
    for items in train_interactions.values():
        train_items.update(items)
    
    test_items = set()
    for items in test_interactions.values():
        test_items.update(items)
    
    print(f"\nData integrity check:")
    print(f"Unique items in train: {len(train_items)}")
    print(f"Unique items in test: {len(test_items)}")
    print(f"Test items also in train: {len(test_items & train_items)} ({100 * len(test_items & train_items) / len(test_items):.1f}%)")
    print(f"Test items NOT in train: {len(test_items - train_items)}")
    
    if len(test_items - train_items) > 0:
        print("WARNING: Some test items are not in training set!")
        print("This should not happen with proper data split.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create proper sample data for KGAT')
    parser.add_argument('--data-dir', type=str, default='data/amazon-book-fixed',
                        help='Directory to save the data')
    parser.add_argument('--n-users', type=int, default=1000,
                        help='Number of users')
    parser.add_argument('--n-items', type=int, default=2000,
                        help='Number of items')
    parser.add_argument('--n-relations', type=int, default=5,
                        help='Number of relation types')
    
    args = parser.parse_args()
    
    create_proper_sample_data(
        args.data_dir,
        n_users=args.n_users,
        n_items=args.n_items,
        n_relations=args.n_relations
    )