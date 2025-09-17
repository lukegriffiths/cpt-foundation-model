#!/usr/bin/env python3
"""
Utility script to analyze and plot MLflow experiment results.
This script can be used to compare multiple training runs and create custom visualizations.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    print("MLflow not installed. Please run: pip install mlflow")
    sys.exit(1)

def load_experiment_data(experiment_name: str = "cpt-foundation-training"):
    """
    Load all runs from an MLflow experiment.
    
    Args:
        experiment_name: Name of the MLflow experiment
        
    Returns:
        DataFrame with run data
    """
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Experiment '{experiment_name}' not found.")
            return None
        
        runs = client.search_runs(experiment.experiment_id)
        
        # Convert to DataFrame
        run_data = []
        for run in runs:
            run_info = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                'status': run.info.status,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'end_time': pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else None,
            }
            
            # Add parameters
            run_info.update({f"param_{k}": v for k, v in run.data.params.items()})
            
            # Add final metrics
            run_info.update({f"metric_{k}": v for k, v in run.data.metrics.items()})
            
            run_data.append(run_info)
        
        df = pd.DataFrame(run_data)
        print(f"Loaded {len(df)} runs from experiment '{experiment_name}'")
        return df
        
    except Exception as e:
        print(f"Error loading experiment data: {e}")
        return None

def plot_run_comparison(df: pd.DataFrame, save_dir: str = "analysis_plots"):
    """
    Create comparison plots for multiple runs.
    
    Args:
        df: DataFrame with run data
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter completed runs
    completed_runs = df[df['status'] == 'FINISHED'].copy()
    
    if len(completed_runs) == 0:
        print("No completed runs found for comparison.")
        return
    
    # Plot 1: Best validation loss comparison
    if 'metric_final_best_val_loss' in completed_runs.columns:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        completed_runs.plot(x='run_name', y='metric_final_best_val_loss', 
                          kind='bar', ax=plt.gca(), color='skyblue')
        plt.title('Best Validation Loss by Run')
        plt.ylabel('Validation Loss')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Plot 2: Training time comparison
        plt.subplot(1, 2, 2)
        if 'metric_total_training_time' in completed_runs.columns:
            completed_runs.plot(x='run_name', y='metric_total_training_time', 
                              kind='bar', ax=plt.gca(), color='lightcoral')
            plt.title('Total Training Time by Run')
            plt.ylabel('Time (seconds)')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'run_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Hyperparameter analysis
    numeric_params = [col for col in completed_runs.columns 
                     if col.startswith('param_') and completed_runs[col].dtype in ['float64', 'int64']]
    
    if len(numeric_params) > 0 and 'metric_final_best_val_loss' in completed_runs.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, param in enumerate(numeric_params[:4]):  # Show first 4 numeric params
            if i < len(axes):
                completed_runs.plot.scatter(x=param, y='metric_final_best_val_loss', 
                                          ax=axes[i], alpha=0.7)
                axes[i].set_title(f'Val Loss vs {param.replace("param_", "")}')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_params), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'hyperparameter_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Comparison plots saved to {save_dir}")

def get_detailed_metrics(run_id: str, experiment_name: str = "cpt-foundation-training"):
    """
    Get detailed metrics history for a specific run.
    
    Args:
        run_id: MLflow run ID
        experiment_name: Name of the experiment
        
    Returns:
        DataFrame with detailed metrics
    """
    client = MlflowClient()
    
    try:
        # Get metric histories
        metrics = ['train_loss', 'val_loss', 'learning_rate', 'mask_ratio', 'gpu_memory_mb']
        
        detailed_data = {}
        for metric in metrics:
            try:
                history = client.get_metric_history(run_id, metric)
                detailed_data[metric] = [(h.step, h.value) for h in history]
            except:
                detailed_data[metric] = []
        
        # Convert to DataFrame
        max_steps = max(len(data) for data in detailed_data.values()) if detailed_data else 0
        
        df_data = {}
        for metric, data in detailed_data.items():
            steps, values = zip(*data) if data else ([], [])
            df_data[f'{metric}_step'] = list(steps) + [None] * (max_steps - len(steps))
            df_data[f'{metric}_value'] = list(values) + [None] * (max_steps - len(values))
        
        return pd.DataFrame(df_data)
        
    except Exception as e:
        print(f"Error getting detailed metrics: {e}")
        return None

def plot_detailed_run_metrics(run_id: str, experiment_name: str = "cpt-foundation-training", 
                            save_dir: str = "analysis_plots"):
    """
    Create detailed plots for a specific run.
    
    Args:
        run_id: MLflow run ID
        experiment_name: Name of the experiment
        save_dir: Directory to save plots
    """
    df = get_detailed_metrics(run_id, experiment_name)
    if df is None:
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if 'train_loss_value' in df.columns:
        axes[0, 0].plot(df['train_loss_step'], df['train_loss_value'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    if 'val_loss_value' in df.columns:
        val_data = df.dropna(subset=['val_loss_value'])
        axes[0, 1].plot(val_data['val_loss_step'], val_data['val_loss_value'], 'r-', linewidth=2)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate_value' in df.columns:
        axes[1, 0].plot(df['learning_rate_step'], df['learning_rate_value'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # GPU memory
    if 'gpu_memory_mb_value' in df.columns:
        gpu_data = df.dropna(subset=['gpu_memory_mb_value'])
        axes[1, 1].plot(gpu_data['gpu_memory_mb_step'], gpu_data['gpu_memory_mb_value'], 'purple', linewidth=2)
        axes[1, 1].set_title('GPU Memory Usage')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Memory (MB)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'detailed_metrics_{run_id[:8]}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detailed metrics plot saved for run {run_id[:8]}")

def main():
    parser = argparse.ArgumentParser(description="Analyze MLflow experiment results")
    parser.add_argument('--experiment', default='cpt-foundation-training', 
                       help='MLflow experiment name')
    parser.add_argument('--run-id', help='Specific run ID for detailed analysis')
    parser.add_argument('--output-dir', default='analysis_plots', 
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    print(f"Analyzing MLflow experiment: {args.experiment}")
    
    if args.run_id:
        # Detailed analysis for specific run
        plot_detailed_run_metrics(args.run_id, args.experiment, args.output_dir)
    else:
        # Compare all runs in experiment
        df = load_experiment_data(args.experiment)
        if df is not None:
            plot_run_comparison(df, args.output_dir)
            
            # Print summary
            print("\nExperiment Summary:")
            print(f"Total runs: {len(df)}")
            completed = df[df['status'] == 'FINISHED']
            print(f"Completed runs: {len(completed)}")
            
            if len(completed) > 0 and 'metric_final_best_val_loss' in completed.columns:
                best_run = completed.loc[completed['metric_final_best_val_loss'].idxmin()]
                print(f"Best run: {best_run['run_name']} (Val Loss: {best_run['metric_final_best_val_loss']:.6f})")

if __name__ == "__main__":
    main()