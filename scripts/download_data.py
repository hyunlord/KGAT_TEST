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
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


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


def prepare_amazon_book():
    """Download and prepare Amazon-Book dataset"""
    # Note: You'll need to replace this with actual dataset URL
    # This is a placeholder structure
    data_dir = 'data/amazon-book'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Amazon-Book dataset preparation")
    print("Please download the dataset manually from:")
    print("https://github.com/xiangwang1223/knowledge_graph_attention_network")
    print(f"And extract it to: {data_dir}")
    
    # Create sample data for testing
    create_sample_data(data_dir)


def prepare_last_fm():
    """Download and prepare Last-FM dataset"""
    data_dir = 'data/last-fm'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Last-FM dataset preparation")
    print("Please download the dataset manually from:")
    print("https://github.com/xiangwang1223/knowledge_graph_attention_network")
    print(f"And extract it to: {data_dir}")
    
    # Create sample data for testing
    create_sample_data(data_dir)


def prepare_yelp2018():
    """Download and prepare Yelp2018 dataset"""
    data_dir = 'data/yelp2018'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Yelp2018 dataset preparation")
    print("Please download the dataset manually from:")
    print("https://github.com/xiangwang1223/knowledge_graph_attention_network")
    print(f"And extract it to: {data_dir}")
    
    # Create sample data for testing
    create_sample_data(data_dir)


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
    print("Files created:")
    print(f"  - train.txt: 100 users with training interactions")
    print(f"  - test.txt: 100 users with test interactions")
    print(f"  - kg_final.txt: Knowledge graph with item relations")


def main():
    parser = argparse.ArgumentParser(description='Download datasets for KGAT')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['amazon-book', 'last-fm', 'yelp2018', 'all'],
                        help='Which dataset to download')
    parser.add_argument('--sample-only', action='store_true',
                        help='Only create sample data for testing')
    
    args = parser.parse_args()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    if args.sample_only:
        datasets = ['amazon-book', 'last-fm', 'yelp2018'] if args.dataset == 'all' else [args.dataset]
        for dataset in datasets:
            data_dir = f'data/{dataset}'
            os.makedirs(data_dir, exist_ok=True)
            create_sample_data(data_dir)
    else:
        if args.dataset == 'amazon-book' or args.dataset == 'all':
            prepare_amazon_book()
        
        if args.dataset == 'last-fm' or args.dataset == 'all':
            prepare_last_fm()
        
        if args.dataset == 'yelp2018' or args.dataset == 'all':
            prepare_yelp2018()
    
    print("\nDataset preparation completed!")
    print("\nTo use a dataset, update the config.yaml file:")
    print("  data.data_dir: data/<dataset-name>")
    print("\nOr set environment variable:")
    print("  export DATA_DIR=data/<dataset-name>")


if __name__ == "__main__":
    main()