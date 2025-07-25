import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

from data_loader import DataLoader
from kgat_model import KGAT
from evaluator import Evaluator


class MethodComparison:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize data loader
        self.data_loader = DataLoader(config['data_path'])
        self.data_loader.load_cf_data()
        self.data_loader.load_kg_data()
        
        # Initialize model
        self.model = KGAT(
            n_users=self.data_loader.n_users,
            n_entities=self.data_loader.n_entities,
            n_relations=self.data_loader.n_relations,
            embed_dim=config['embed_dim'],
            layer_dims=config['layer_dims'],
            aggregator=config['aggregator']
        ).to(self.device)
        
        # Load pretrained weights if available
        if 'model_path' in config and os.path.exists(config['model_path']):
            self.model.load_state_dict(torch.load(config['model_path']))
            print(f"Loaded model from {config['model_path']}")
        
        # Initialize evaluator
        self.evaluator = Evaluator(
            self.model, 
            self.data_loader, 
            self.device,
            top_k=config.get('top_k', [20, 40, 60])
        )
        
        # Prepare graph data
        self.edge_index_ui = self._prepare_edge_index_ui()
        self.edge_index_kg, self.edge_type_kg = self._prepare_edge_index_kg()
        
    def _prepare_edge_index_ui(self):
        """Prepare edge index for user-item bipartite graph"""
        edge_list = []
        
        for user, items in self.data_loader.train_data.items():
            for item in items:
                edge_list.append([user, self.data_loader.n_users + item])
                edge_list.append([self.data_loader.n_users + item, user])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
        return edge_index
    
    def _prepare_edge_index_kg(self):
        """Prepare edge index and edge types for knowledge graph"""
        edge_list = []
        edge_types = []
        
        for head, relation, tail in self.data_loader.kg_data:
            if head < self.data_loader.n_items and tail < self.data_loader.n_items:
                edge_list.append([head, tail])
                edge_types.append(relation)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
            edge_type = torch.tensor(edge_types, dtype=torch.long).to(self.device)
            return edge_index, edge_type
        else:
            return None, None
    
    def compare_methods(self):
        """Compare standard and relation-enhanced recommendation methods"""
        print("Evaluating standard method (user-item similarity only)...")
        standard_metrics = self.evaluator.evaluate_standard(
            self.edge_index_ui, self.edge_index_kg, self.edge_type_kg
        )
        
        print("\nEvaluating enhanced method (user+relation-item similarity)...")
        enhanced_metrics = self.evaluator.evaluate_with_relations(
            self.edge_index_ui, self.edge_index_kg, self.edge_type_kg
        )
        
        # Calculate improvement
        improvements = {}
        for metric in standard_metrics:
            improvements[metric] = {}
            for k in standard_metrics[metric]:
                std_val = standard_metrics[metric][k]
                enh_val = enhanced_metrics[metric][k]
                improvements[metric][k] = (enh_val - std_val) / std_val * 100 if std_val > 0 else 0
        
        return {
            'standard': standard_metrics,
            'enhanced': enhanced_metrics,
            'improvements': improvements
        }
    
    def visualize_results(self, results, save_path='results'):
        """Create visualizations comparing the two methods"""
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Bar plot comparing metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['recall', 'precision', 'ndcg']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            k_values = list(results['standard'][metric].keys())
            standard_values = [results['standard'][metric][k] for k in k_values]
            enhanced_values = [results['enhanced'][metric][k] for k in k_values]
            
            x = np.arange(len(k_values))
            width = 0.35
            
            ax.bar(x - width/2, standard_values, width, label='Standard', alpha=0.8)
            ax.bar(x + width/2, enhanced_values, width, label='Enhanced', alpha=0.8)
            
            ax.set_xlabel('K')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([f'@{k}' for k in k_values])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'metrics_comparison.png'))
        plt.close()
        
        # 2. Improvement heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        improvement_data = []
        for metric in metrics:
            for k in results['improvements'][metric]:
                improvement_data.append({
                    'Metric': metric.upper(),
                    'K': f'@{k}',
                    'Improvement (%)': results['improvements'][metric][k]
                })
        
        df_improvements = pd.DataFrame(improvement_data)
        pivot_df = df_improvements.pivot(index='Metric', columns='K', values='Improvement (%)')
        
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, 
                   cbar_kws={'label': 'Improvement (%)'}, ax=ax)
        ax.set_title('Performance Improvement: Enhanced vs Standard Method')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'improvement_heatmap.png'))
        plt.close()
        
        # 3. Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(save_path, f'comparison_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {results_file}")
        print(f"Visualizations saved to {save_path}/")
        
    def analyze_sample_users(self, n_users=5):
        """Analyze recommendations for sample users"""
        sample_users = np.random.choice(
            list(self.data_loader.test_data.keys()), 
            size=min(n_users, len(self.data_loader.test_data)), 
            replace=False
        )
        
        analyses = []
        for user_id in sample_users:
            analysis = self.evaluator.analyze_recommendations(
                user_id, self.edge_index_ui, self.edge_index_kg, self.edge_type_kg
            )
            analyses.append(analysis)
        
        return analyses
    
    def print_summary(self, results):
        """Print a summary of the comparison results"""
        print("\n" + "="*60)
        print("RECOMMENDATION METHOD COMPARISON SUMMARY")
        print("="*60)
        
        print("\nStandard Method (User-Item Similarity Only):")
        for metric in results['standard']:
            print(f"\n{metric.upper()}:")
            for k, value in results['standard'][metric].items():
                print(f"  @{k}: {value:.4f}")
        
        print("\n\nEnhanced Method (User+Relation-Item Similarity):")
        for metric in results['enhanced']:
            print(f"\n{metric.upper()}:")
            for k, value in results['enhanced'][metric].items():
                print(f"  @{k}: {value:.4f}")
        
        print("\n\nImprovement (%):")
        for metric in results['improvements']:
            print(f"\n{metric.upper()}:")
            for k, value in results['improvements'][metric].items():
                print(f"  @{k}: {value:+.2f}%")
        
        print("\n" + "="*60)


def main():
    # Configuration
    config = {
        'data_path': 'data/',  # Path to your dataset
        'embed_dim': 64,
        'layer_dims': [32, 16],
        'aggregator': 'bi-interaction',
        'top_k': [10, 20, 30, 50],
        'model_path': 'model/kgat_model.pth'  # Optional: path to pretrained model
    }
    
    # Run comparison
    comparison = MethodComparison(config)
    results = comparison.compare_methods()
    
    # Print summary
    comparison.print_summary(results)
    
    # Visualize results
    comparison.visualize_results(results)
    
    # Analyze sample users
    print("\nAnalyzing sample users...")
    sample_analyses = comparison.analyze_sample_users(n_users=3)
    
    for analysis in sample_analyses:
        print(f"\nUser {analysis['user_id']}:")
        print(f"  Test items: {len(analysis['test_items'])}")
        print(f"  Standard hits: {len(analysis['standard_hits'])}")
        print(f"  Enhanced hits: {len(analysis['enhanced_hits'])}")
        print(f"  Recommendation overlap: {analysis['overlap']}/20")


if __name__ == "__main__":
    main()