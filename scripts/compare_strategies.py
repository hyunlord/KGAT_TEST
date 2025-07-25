#!/usr/bin/env python3
"""
Compare training strategies: DDP vs DeepSpeed
This script runs short training sessions with different strategies and compares performance
"""

import os
import sys
import time
import subprocess
import argparse
import json
from datetime import datetime
import psutil
import GPUtil


def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    gpus = GPUtil.getGPUs()
    memory_usage = []
    for gpu in gpus:
        memory_usage.append({
            'id': gpu.id,
            'name': gpu.name,
            'memory_used': gpu.memoryUsed,
            'memory_total': gpu.memoryTotal,
            'memory_util': gpu.memoryUtil * 100
        })
    return memory_usage


def run_training(strategy, config_overrides, max_epochs=10):
    """Run training with specified strategy"""
    base_cmd = [
        'python', 'src/train.py',
        '--config-name', 'config_multi_gpu',
        f'training.max_epochs={max_epochs}',
        f'training.strategy={strategy}',
        'training.check_val_every_n_epoch=2'
    ]
    
    # Add config overrides
    for key, value in config_overrides.items():
        base_cmd.append(f'{key}={value}')
    
    # Record start time and memory
    start_time = time.time()
    start_memory = get_gpu_memory_usage()
    
    print(f"\n{'='*60}")
    print(f"Running training with strategy: {strategy}")
    print(f"Command: {' '.join(base_cmd)}")
    print(f"{'='*60}\n")
    
    # Run training
    try:
        result = subprocess.run(
            base_cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Get final memory usage
        end_memory = get_gpu_memory_usage()
        
        # Parse output for metrics
        output_lines = result.stdout.split('\n')
        metrics = parse_training_output(output_lines)
        
        return {
            'strategy': strategy,
            'success': True,
            'training_time': training_time,
            'start_memory': start_memory,
            'end_memory': end_memory,
            'metrics': metrics,
            'config': config_overrides
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Error running training: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        return {
            'strategy': strategy,
            'success': False,
            'error': str(e),
            'stdout': e.stdout,
            'stderr': e.stderr
        }


def parse_training_output(output_lines):
    """Parse training output for metrics"""
    metrics = {
        'final_train_loss': None,
        'final_val_recall@20': None,
        'epoch_times': [],
        'gpu_memory_peaks': []
    }
    
    for line in output_lines:
        # Look for training metrics
        if 'train_loss' in line and 'Step' in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'train_loss:':
                        metrics['final_train_loss'] = float(parts[i+1])
            except:
                pass
        
        # Look for validation metrics
        if 'val_recall@20' in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith('val_recall@20'):
                        value = part.split(':')[1] if ':' in part else parts[i+1]
                        metrics['final_val_recall@20'] = float(value)
            except:
                pass
    
    return metrics


def compare_strategies(args):
    """Compare different training strategies"""
    results = []
    
    # Common configuration
    base_config = {
        'data.data_dir': args.data_dir,
        'data.batch_size': args.batch_size,
        'training.devices': args.devices,
        'training.precision': 16
    }
    
    # Test DDP
    if 'ddp' in args.strategies:
        print("\n" + "="*80)
        print("Testing DDP (Distributed Data Parallel)")
        print("="*80)
        
        ddp_config = base_config.copy()
        ddp_result = run_training('ddp', ddp_config, args.max_epochs)
        results.append(ddp_result)
        
        # Cool down between tests
        time.sleep(10)
    
    # Test DeepSpeed Stage 1
    if 'deepspeed_stage_1' in args.strategies:
        print("\n" + "="*80)
        print("Testing DeepSpeed Stage 1")
        print("="*80)
        
        ds1_config = base_config.copy()
        ds1_result = run_training('deepspeed_stage_1', ds1_config, args.max_epochs)
        results.append(ds1_result)
        
        time.sleep(10)
    
    # Test DeepSpeed Stage 2
    if 'deepspeed_stage_2' in args.strategies:
        print("\n" + "="*80)
        print("Testing DeepSpeed Stage 2")
        print("="*80)
        
        ds2_config = base_config.copy()
        ds2_result = run_training('deepspeed_stage_2', ds2_config, args.max_epochs)
        results.append(ds2_result)
        
        time.sleep(10)
    
    # Test DeepSpeed Stage 3 (if enough memory)
    if 'deepspeed_stage_3' in args.strategies and args.test_stage_3:
        print("\n" + "="*80)
        print("Testing DeepSpeed Stage 3")
        print("="*80)
        
        ds3_config = base_config.copy()
        ds3_config['data.batch_size'] = args.batch_size // 2  # Reduce batch size for Stage 3
        ds3_result = run_training('deepspeed_stage_3', ds3_config, args.max_epochs)
        results.append(ds3_result)
    
    return results


def print_comparison_report(results):
    """Print comparison report"""
    print("\n" + "="*80)
    print("TRAINING STRATEGY COMPARISON REPORT")
    print("="*80)
    
    # Summary table
    print("\n## Performance Summary\n")
    print(f"{'Strategy':<20} {'Success':<10} {'Time (s)':<15} {'Train Loss':<15} {'Val Recall@20':<15}")
    print("-" * 80)
    
    for result in results:
        if result['success']:
            strategy = result['strategy']
            time_taken = result['training_time']
            train_loss = result['metrics'].get('final_train_loss', 'N/A')
            val_recall = result['metrics'].get('final_val_recall@20', 'N/A')
            
            print(f"{strategy:<20} {'Yes':<10} {time_taken:<15.2f} {train_loss:<15} {val_recall:<15}")
        else:
            print(f"{result['strategy']:<20} {'No':<10} {'Failed':<15} {'N/A':<15} {'N/A':<15}")
    
    # Memory usage comparison
    print("\n## GPU Memory Usage\n")
    print(f"{'Strategy':<20} {'Avg Memory (MB)':<20} {'Peak Memory (MB)':<20}")
    print("-" * 60)
    
    for result in results:
        if result['success'] and result.get('end_memory'):
            strategy = result['strategy']
            avg_memory = sum(gpu['memory_used'] for gpu in result['end_memory']) / len(result['end_memory'])
            peak_memory = max(gpu['memory_used'] for gpu in result['end_memory'])
            
            print(f"{strategy:<20} {avg_memory:<20.2f} {peak_memory:<20.2f}")
    
    # Speed comparison
    if len([r for r in results if r['success']]) > 1:
        print("\n## Speed Comparison (relative to DDP)\n")
        
        ddp_time = next((r['training_time'] for r in results if r['strategy'] == 'ddp' and r['success']), None)
        if ddp_time:
            for result in results:
                if result['success']:
                    speedup = ddp_time / result['training_time']
                    print(f"{result['strategy']}: {speedup:.2f}x")
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f'results/strategy_comparison_{timestamp}.json'
    os.makedirs('results', exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Compare training strategies for KGAT')
    parser.add_argument('--data-dir', type=str, default='data/amazon-book',
                        help='Path to dataset directory')
    parser.add_argument('--devices', type=int, default=-1,
                        help='Number of GPUs to use (-1 for all)')
    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Batch size for training')
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Number of epochs for comparison')
    parser.add_argument('--strategies', nargs='+', 
                        default=['ddp', 'deepspeed_stage_1', 'deepspeed_stage_2', 'deepspeed_stage_3'],
                        choices=['ddp', 'deepspeed_stage_1', 'deepspeed_stage_2', 'deepspeed_stage_3'],
                        help='Strategies to compare')
    parser.add_argument('--test-stage-3', action='store_true',
                        help='Also test DeepSpeed Stage 3 (requires more memory)')
    
    args = parser.parse_args()
    
    # Check if deepspeed is installed
    try:
        import deepspeed
        print(f"DeepSpeed version: {deepspeed.__version__}")
    except ImportError:
        print("Warning: DeepSpeed not installed. Install with: pip install deepspeed")
        print("Continuing with DDP only...")
        args.strategies = ['ddp']
    
    # Check GPUtil
    try:
        import GPUtil
    except ImportError:
        print("Warning: GPUtil not installed. Install with: pip install gputil")
        print("GPU memory monitoring will be limited.")
    
    # Run comparison
    print(f"\nComparing strategies: {args.strategies}")
    print(f"Dataset: {args.data_dir}")
    print(f"Devices: {args.devices}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    
    results = compare_strategies(args)
    
    # Print report
    print_comparison_report(results)


if __name__ == "__main__":
    main()