"""
Script to compare original KGAT with improved KGAT implementation
"""
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import time

from kgat_lightning import KGATLightning
from kgat_improved import KGATImproved
from data_module import KGATDataModule


def evaluate_model(model, data_module, model_name):
    """Evaluate a model and return metrics"""
    print(f"\nEvaluating {model_name}...")
    
    # Create a simple trainer for evaluation
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False
    )
    
    # Time the evaluation
    start_time = time.time()
    
    # Test the model
    test_results = trainer.test(model, data_module, verbose=False)
    
    eval_time = time.time() - start_time
    
    # Extract metrics
    metrics = test_results[0]
    metrics['eval_time'] = eval_time
    metrics['model_name'] = model_name
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics['total_params'] = total_params
    metrics['trainable_params'] = trainable_params
    
    return metrics


def train_and_evaluate(cfg: DictConfig, model_class, model_name, max_epochs=50):
    """Train a model for a few epochs and evaluate"""
    # Create data module
    data_module = KGATDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        neg_sample_size=cfg.data.neg_sample_size
    )
    
    # Setup data
    data_module.setup()
    
    # Update model config
    cfg.model.n_users = data_module.n_users
    cfg.model.n_entities = data_module.n_entities
    cfg.model.n_relations = data_module.n_relations
    
    # Create model
    model = model_class(cfg.model)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        check_val_every_n_epoch=5
    )
    
    print(f"\nTraining {model_name} for {max_epochs} epochs...")
    start_time = time.time()
    
    # Train
    trainer.fit(model, data_module)
    
    train_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_model(model, data_module, model_name)
    metrics['train_time'] = train_time
    metrics['epochs_trained'] = trainer.current_epoch
    
    return metrics, model


def compare_aggregators(cfg: DictConfig):
    """Compare different aggregator types"""
    aggregators = ['bi-interaction', 'gcn', 'graphsage']
    results = []
    
    for aggregator in aggregators:
        print(f"\n{'='*50}")
        print(f"Testing Improved KGAT with {aggregator} aggregator")
        print(f"{'='*50}")
        
        # Update config
        cfg.model.aggregator = aggregator
        
        # Train and evaluate
        metrics, _ = train_and_evaluate(
            cfg, 
            KGATImproved, 
            f"KGAT-Improved-{aggregator}",
            max_epochs=30
        )
        
        results.append(metrics)
    
    return results


def plot_comparison(results: List[Dict], save_path: str = "model_comparison.png"):
    """Create comparison plots"""
    df = pd.DataFrame(results)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('KGAT Model Comparison', fontsize=16)
    
    # 1. Performance metrics comparison
    ax = axes[0, 0]
    metrics_to_plot = ['test_recall@20', 'test_precision@20', 'test_ndcg@20']
    df_metrics = df[['model_name'] + metrics_to_plot].set_index('model_name')
    df_metrics.plot(kind='bar', ax=ax)
    ax.set_title('Performance Metrics Comparison')
    ax.set_ylabel('Score')
    ax.legend(['Recall@20', 'Precision@20', 'NDCG@20'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Training time comparison
    ax = axes[0, 1]
    df[['model_name', 'train_time']].set_index('model_name').plot(kind='bar', ax=ax, legend=False)
    ax.set_title('Training Time Comparison')
    ax.set_ylabel('Time (seconds)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Model size comparison
    ax = axes[1, 0]
    df[['model_name', 'trainable_params']].set_index('model_name').plot(kind='bar', ax=ax, legend=False)
    ax.set_title('Model Size (Trainable Parameters)')
    ax.set_ylabel('Number of Parameters')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 4. Recall@K for different K values
    ax = axes[1, 1]
    recall_cols = [col for col in df.columns if 'recall@' in col and 'test_' in col]
    df_recall = df[['model_name'] + recall_cols].set_index('model_name')
    df_recall.columns = [col.replace('test_recall@', 'K=') for col in df_recall.columns]
    df_recall.plot(ax=ax)
    ax.set_title('Recall at Different K Values')
    ax.set_ylabel('Recall')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {save_path}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """Main comparison function"""
    # Set random seed
    seed_everything(cfg.training.seed)
    
    # Convert to absolute path
    if not os.path.isabs(cfg.data.data_dir):
        cfg.data.data_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.data.data_dir)
    
    print("\n" + "="*70)
    print("KGAT Model Comparison: Original vs Improved Implementation")
    print("="*70)
    
    results = []
    
    # 1. Test original KGAT
    print(f"\n{'='*50}")
    print("Testing Original KGAT")
    print(f"{'='*50}")
    
    cfg.model.aggregator = 'bi-interaction'
    metrics_original, model_original = train_and_evaluate(
        cfg, 
        KGATLightning, 
        "KGAT-Original",
        max_epochs=30
    )
    results.append(metrics_original)
    
    # 2. Test improved KGAT with same aggregator
    print(f"\n{'='*50}")
    print("Testing Improved KGAT (bi-interaction)")
    print(f"{'='*50}")
    
    metrics_improved, model_improved = train_and_evaluate(
        cfg, 
        KGATImproved, 
        "KGAT-Improved-BI",
        max_epochs=30
    )
    results.append(metrics_improved)
    
    # 3. Test different aggregators (optional)
    if cfg.get('test_all_aggregators', False):
        aggregator_results = compare_aggregators(cfg)
        results.extend(aggregator_results)
    
    # Create comparison table
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    
    # Display key metrics
    display_columns = [
        'model_name', 
        'test_recall@20', 
        'test_precision@20', 
        'test_ndcg@20',
        'train_time',
        'trainable_params'
    ]
    
    print("\nKey Metrics Comparison:")
    print(df_results[display_columns].to_string(index=False))
    
    # Calculate improvements
    print("\n" + "="*70)
    print("IMPROVEMENT ANALYSIS")
    print("="*70)
    
    original_recall = metrics_original['test_recall@20']
    improved_recall = metrics_improved['test_recall@20']
    improvement = ((improved_recall - original_recall) / original_recall) * 100
    
    print(f"\nRecall@20 Improvement: {improvement:.2f}%")
    print(f"Original: {original_recall:.4f}")
    print(f"Improved: {improved_recall:.4f}")
    
    # Save detailed results
    results_path = "model_comparison_results.csv"
    df_results.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")
    
    # Create visualization
    plot_comparison(results, "model_comparison.png")
    
    # Key differences summary
    print("\n" + "="*70)
    print("KEY DIFFERENCES IMPLEMENTED")
    print("="*70)
    print("1. ✓ Layer output concatenation (including initial embedding)")
    print("2. ✓ L2 normalization after each layer")
    print("3. ✓ Multiple aggregation types (bi-interaction, GCN, GraphSAGE)")
    print("4. ✓ Improved attention mechanism")
    print("5. ✓ Edge normalization for GCN aggregator")
    print("6. ✓ Final transformation layer for dimension reduction")
    
    return df_results


if __name__ == "__main__":
    main()