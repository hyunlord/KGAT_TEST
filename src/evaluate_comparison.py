"""
Compare standard and relation-enhanced recommendation methods using trained KGAT model
"""

import os
import torch
import argparse
import json
from datetime import datetime
from omegaconf import OmegaConf

from kgat_lightning import KGATLightning
from data_module import KGATDataModule
from evaluator import Evaluator
from compare_methods import MethodComparison


def load_trained_model(checkpoint_path, config_path):
    """Load trained KGAT model from checkpoint"""
    # Load config
    config = OmegaConf.load(config_path)
    
    # Initialize data module to get statistics
    data_module = KGATDataModule(config.data)
    data_module.setup()
    
    # Update model config with data statistics
    stats = data_module.get_statistics()
    config.model.n_users = stats['n_users']
    config.model.n_entities = stats['n_entities']
    config.model.n_relations = stats['n_relations']
    
    # Load model with compatibility handling
    try:
        # Try loading normally
        model = KGATLightning.load_from_checkpoint(
            checkpoint_path,
            config=config.model,
            map_location='cpu'
        )
    except RuntimeError as e:
        if "relation_embedding" in str(e):
            # Handle old checkpoints with relation_embedding
            print("검출된 이전 버전 체크포인트, 호환성 모드로 로딩...")
            model = KGATLightning(config.model)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['state_dict']
            
            # Remove relation_embedding from state dict if present
            state_dict = {k: v for k, v in state_dict.items() 
                         if 'relation_embedding' not in k}
            
            # Load the filtered state dict
            model.load_state_dict(state_dict, strict=False)
        else:
            raise e
    
    return model, data_module


def main():
    parser = argparse.ArgumentParser(description='Compare KGAT recommendation methods')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='results/comparison',
                        help='Directory to save results')
    parser.add_argument('--n-sample-users', type=int, default=10,
                        help='Number of sample users to analyze')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading trained model...")
    model, data_module = load_trained_model(args.checkpoint, args.config)
    
    # Initialize comparison
    comparison_config = {
        'data_path': data_module.data_dir,
        'embed_dim': model.embed_dim,
        'layer_dims': model.layer_dims,
        'aggregator': model.aggregator,
        'top_k': [10, 20, 30, 50],
        'model': model,  # Use loaded model directly
        'data_loader': data_module
    }
    
    comparison = MethodComparison(comparison_config)
    comparison.model = model  # Use the trained model
    comparison.data_loader = data_module
    
    # Prepare edge indices
    comparison.edge_index_ui = data_module.edge_index_ui.to(model.device)
    comparison.edge_index_kg = data_module.edge_index_kg.to(model.device) if data_module.edge_index_kg is not None else None
    comparison.edge_type_kg = data_module.edge_type_kg.to(model.device) if data_module.edge_type_kg is not None else None
    
    print("\nComparing recommendation methods...")
    results = comparison.compare_methods()
    
    # Print summary
    comparison.print_summary(results)
    
    # Visualize results
    comparison.visualize_results(results, save_path=args.output_dir)
    
    # Analyze sample users
    print(f"\nAnalyzing {args.n_sample_users} sample users...")
    sample_analyses = comparison.analyze_sample_users(n_users=args.n_sample_users)
    
    # Save detailed analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    analysis_file = os.path.join(args.output_dir, f'user_analysis_{timestamp}.json')
    
    analysis_results = {
        'comparison_results': results,
        'sample_user_analyses': sample_analyses,
        'config': {
            'checkpoint': args.checkpoint,
            'n_sample_users': args.n_sample_users
        }
    }
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {analysis_file}")
    
    # Print sample user analysis
    print("\nSample User Analysis:")
    print("-" * 60)
    
    for analysis in sample_analyses[:5]:  # Show first 5 users
        user_id = analysis['user_id']
        print(f"\nUser {user_id}:")
        print(f"  Ground truth items: {len(analysis['test_items'])}")
        print(f"  Standard method hits: {len(analysis['standard_hits'])} / {len(analysis['standard_recommendations'])}")
        print(f"  Enhanced method hits: {len(analysis['enhanced_hits'])} / {len(analysis['enhanced_recommendations'])}")
        print(f"  Recommendation overlap: {analysis['overlap']} items")
        
        if len(analysis['enhanced_hits']) > len(analysis['standard_hits']):
            print(f"  ✓ Enhanced method performs better (+{len(analysis['enhanced_hits']) - len(analysis['standard_hits'])} hits)")
        elif len(analysis['enhanced_hits']) < len(analysis['standard_hits']):
            print(f"  ✗ Standard method performs better (+{len(analysis['standard_hits']) - len(analysis['enhanced_hits'])} hits)")
        else:
            print(f"  = Both methods perform equally")


if __name__ == "__main__":
    main()