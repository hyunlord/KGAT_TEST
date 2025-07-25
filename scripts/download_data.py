#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for KGAT
Supports: Amazon-Book, Last-FM, Yelp2018
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
import gdown
from tqdm import tqdm


# Dataset URLs from KGAT repository
DATASET_URLS = {
    'amazon-book': {
        'url': 'https://drive.google.com/uc?id=1qQxBE3N8Cx6KsLmQ8tqPceE8_5dYGLe2',
        'file': 'amazon-book.zip',
        'extract_dir': 'amazon-book'
    },
    'last-fm': {
        'url': 'https://drive.google.com/uc?id=1GJW-4b0HqsXp1bVALKf7kdGwJ-JLPeNB',
        'file': 'last-fm.zip', 
        'extract_dir': 'last-fm'
    },
    'yelp2018': {
        'url': 'https://drive.google.com/uc?id=1YGrI1q3uHaVuMEzE5hXJ23c1_jC-O9Dq',
        'file': 'yelp2018.zip',
        'extract_dir': 'yelp2018'
    }
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_from_google_drive(url, output_path):
    """Download file from Google Drive"""
    print(f"Downloading {output_path}...")
    try:
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(file_path, extract_to='.'):
    """Extract zip or tar.gz archive"""
    print(f"Extracting {file_path}...")
    
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported archive format: {file_path}")


def verify_dataset_files(data_dir):
    """Verify that required files exist in the dataset directory"""
    required_files = ['train.txt', 'test.txt', 'kg_final.txt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Missing required files in {data_dir}: {missing_files}")
        return False
    
    return True


def download_dataset(dataset_name):
    """Download and extract a specific dataset"""
    if dataset_name not in DATASET_URLS:
        print(f"Unknown dataset: {dataset_name}")
        return False
    
    dataset_info = DATASET_URLS[dataset_name]
    data_dir = os.path.join('data', dataset_name)
    
    # Check if dataset already exists
    if os.path.exists(data_dir) and verify_dataset_files(data_dir):
        print(f"Dataset {dataset_name} already exists and is complete.")
        return True
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download dataset
    zip_path = os.path.join('data', dataset_info['file'])
    
    print(f"\nDownloading {dataset_name} dataset...")
    success = download_from_google_drive(dataset_info['url'], zip_path)
    
    if not success:
        print(f"Failed to download {dataset_name}. Trying alternative method...")
        # Alternative: provide manual download instructions
        print(f"\nPlease download manually from:")
        print(f"URL: {dataset_info['url']}")
        print(f"Save to: {zip_path}")
        return False
    
    # Extract dataset
    try:
        extract_archive(zip_path, 'data')
        
        # Clean up zip file
        os.remove(zip_path)
        
        # Verify extraction
        if verify_dataset_files(data_dir):
            print(f"Successfully downloaded and extracted {dataset_name}")
            
            # Print dataset statistics
            print_dataset_stats(data_dir)
            return True
        else:
            print(f"Dataset extraction may have failed. Please check {data_dir}")
            return False
            
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return False


def print_dataset_stats(data_dir):
    """Print basic statistics about the dataset"""
    print(f"\nDataset statistics for {data_dir}:")
    
    # Count users and items in train.txt
    train_file = os.path.join(data_dir, 'train.txt')
    if os.path.exists(train_file):
        users = set()
        items = set()
        interactions = 0
        
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    user = int(parts[0])
                    user_items = [int(item) for item in parts[1:]]
                    users.add(user)
                    items.update(user_items)
                    interactions += len(user_items)
        
        print(f"  Training: {len(users)} users, {len(items)} items, {interactions} interactions")
    
    # Count test interactions
    test_file = os.path.join(data_dir, 'test.txt')
    if os.path.exists(test_file):
        test_users = 0
        test_interactions = 0
        
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    test_users += 1
                    test_interactions += len(parts) - 1
        
        print(f"  Test: {test_users} users, {test_interactions} interactions")
    
    # Count KG triples
    kg_file = os.path.join(data_dir, 'kg_final.txt')
    if os.path.exists(kg_file):
        kg_triples = 0
        relations = set()
        
        with open(kg_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    kg_triples += 1
                    relations.add(int(parts[1]))
        
        print(f"  KG: {kg_triples} triples, {len(relations)} relation types")


def create_sample_data(data_dir):
    """Create sample data for testing"""
    print(f"\nCreating sample data in {data_dir} for testing...")
    
    # Create train.txt - user-item interactions
    with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
        # Format: user_id item_id1 item_id2 ...
        for user in range(100):
            items = []
            # Each user interacts with 5-20 items
            n_items = 5 + user % 15
            for i in range(n_items):
                item = (user * 7 + i * 3) % 500  # Generate pseudo-random items
                items.append(str(item))
            f.write(f"{user} {' '.join(items)}\n")
    
    # Create test.txt
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for user in range(100):
            items = []
            # Each user has 1-5 test items
            n_items = 1 + user % 5
            for i in range(n_items):
                item = (user * 11 + i * 7 + 500) % 1000  # Different items from train
                items.append(str(item))
            f.write(f"{user} {' '.join(items)}\n")
    
    # Create kg_final.txt - knowledge graph triples
    with open(os.path.join(data_dir, 'kg_final.txt'), 'w') as f:
        # Format: head_entity relation_id tail_entity
        # Create some item-item relations
        for i in range(500):
            # Each item has 2-10 relations
            n_relations = 2 + i % 8
            for j in range(n_relations):
                head = i
                relation = j % 5  # 5 different relation types
                tail = (i + j * 13 + 1) % 500  # Related item
                f.write(f"{head} {relation} {tail}\n")
    
    print(f"Sample data created successfully in {data_dir}")
    print_dataset_stats(data_dir)


def main():
    parser = argparse.ArgumentParser(description='Download datasets for KGAT')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['amazon-book', 'last-fm', 'yelp2018', 'all'],
                        help='Which dataset to download')
    parser.add_argument('--sample-only', action='store_true',
                        help='Only create sample data for testing')
    
    args = parser.parse_args()
    
    # Check if gdown is installed
    try:
        import gdown
    except ImportError:
        print("Error: gdown is required for downloading from Google Drive")
        print("Please install it with: pip install gdown")
        sys.exit(1)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    if args.sample_only:
        datasets = ['amazon-book', 'last-fm', 'yelp2018'] if args.dataset == 'all' else [args.dataset]
        for dataset in datasets:
            data_dir = f'data/{dataset}'
            os.makedirs(data_dir, exist_ok=True)
            create_sample_data(data_dir)
    else:
        # Download real datasets
        datasets = ['amazon-book', 'last-fm', 'yelp2018'] if args.dataset == 'all' else [args.dataset]
        
        success_count = 0
        for dataset in datasets:
            if download_dataset(dataset):
                success_count += 1
            print()  # Empty line between datasets
        
        print(f"\nDataset download completed! ({success_count}/{len(datasets)} successful)")
        
        if success_count > 0:
            print("\nTo use a dataset, update the config.yaml file:")
            print("  data.data_dir: data/<dataset-name>")
            print("\nOr set environment variable:")
            print("  export DATA_DIR=data/<dataset-name>")


if __name__ == "__main__":
    main()