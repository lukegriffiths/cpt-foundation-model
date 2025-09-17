"""
MLflow-based metrics tracking for CPT Foundation Model training.
Provides comprehensive experiment tracking, metrics logging, and model management.
"""
import os
import json
import time
import warnings
from typing import Dict, List, Any, Optional

# Suppress specific MLflow warnings
warnings.filterwarnings("ignore", message=".*Found torch version.*contains a local version label.*")
warnings.filterwarnings("ignore", message=".*artifact_path.*is deprecated.*")

import mlflow
import mlflow.pytorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CPTMetricsTracker:
    """
    Comprehensive metrics tracking for CPT model training using MLflow.
    """
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None, 
                 tracking_uri: Optional[str] = None):
        """
        Initialize the metrics tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            run_name: Name of this specific run (auto-generated if None)
            tracking_uri: MLflow tracking server URI (uses local ./mlruns if None)
        """
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Set tracking URI (local by default)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local mlruns directory
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Set or create experiment
        mlflow.set_experiment(experiment_name)
        
        # Initialize tracking variables
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'mask_ratio': [],
            'epoch_time': [],
            'gpu_memory': []
        }
        
        self.start_time = None
        self.current_epoch = 0
        
    def start_run(self, config: Dict[str, Any]):
        """
        Start MLflow run and log configuration.
        
        Args:
            config: Training configuration dictionary
        """
        self.start_time = time.time()
        
        # Start MLflow run
        mlflow.start_run(run_name=self.run_name)
        
        # Log all configuration parameters
        self._log_config(config)
        
        print(f"Started MLflow run: {self.run_name}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        
    def _log_config(self, config: Dict[str, Any]):
        """Log configuration parameters to MLflow."""
        # Flatten nested config
        flat_config = self._flatten_dict(config)
        
        # Log parameters
        for key, value in flat_config.items():
            mlflow.log_param(key, value)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary for MLflow parameter logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_epoch_metrics(self, epoch: int, train_loss: float, 
                         learning_rate: float, mask_ratio: float,
                         val_loss: Optional[float] = None,
                         epoch_time: Optional[float] = None,
                         gpu_memory_mb: Optional[float] = None):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss for this epoch
            learning_rate: Current learning rate
            mask_ratio: Actual masking ratio used
            val_loss: Validation loss (if available)
            epoch_time: Time taken for this epoch in seconds
            gpu_memory_mb: GPU memory usage in MB
        """
        self.current_epoch = epoch
        
        # Log to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("learning_rate", learning_rate, step=epoch)
        mlflow.log_metric("mask_ratio", mask_ratio, step=epoch)
        
        if val_loss is not None:
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            
        if epoch_time is not None:
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)
            
        if gpu_memory_mb is not None:
            mlflow.log_metric("gpu_memory_mb", gpu_memory_mb, step=epoch)
        
        # Store in history for plotting
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['learning_rate'].append(learning_rate)
        self.metrics_history['mask_ratio'].append(mask_ratio)
        
        if val_loss is not None:
            self.metrics_history['val_loss'].append(val_loss)
        if epoch_time is not None:
            self.metrics_history['epoch_time'].append(epoch_time)
        if gpu_memory_mb is not None:
            self.metrics_history['gpu_memory'].append(gpu_memory_mb)
    
    def log_model_checkpoint(self, model: torch.nn.Module, 
                           checkpoint_path: str, 
                           is_best: bool = False):
        """
        Log model checkpoint to MLflow.
        
        Args:
            model: PyTorch model to save
            checkpoint_path: Local path where checkpoint is saved
            is_best: Whether this is the best model so far
        """
        # Log the model file as artifact
        mlflow.log_artifact(checkpoint_path, "checkpoints")
        
        # If this is the best model, also register it (simplified to avoid warnings)
        if is_best:
            try:
                mlflow.pytorch.log_model(
                    model, 
                    artifact_path="best_model",
                    registered_model_name=f"{self.experiment_name}_best"
                )
            except Exception as e:
                print(f"Warning: Could not log model: {e}")
            
            mlflow.log_metric("best_model_epoch", self.current_epoch)
    
    def create_training_plots(self, save_dir: str = "plots"):
        """
        Create and save training plots, also log them to MLflow.
        
        Args:
            save_dir: Directory to save plot files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss plot
        self._plot_losses(save_dir)
        
        # Learning rate plot
        self._plot_learning_rate(save_dir)
        
        print(f"Training plots saved to {save_dir}")
    
    def _plot_losses(self, save_dir: str):
        """Plot training and validation losses."""
        plt.figure(figsize=(12, 6))
        
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if self.metrics_history['val_loss']:
            # Validation is logged every 5 epochs
            val_epochs = range(5, len(self.metrics_history['train_loss']) + 1, 5)
            if len(val_epochs) <= len(self.metrics_history['val_loss']):
                plt.plot(val_epochs[:len(self.metrics_history['val_loss'])], 
                        self.metrics_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(epochs, self.metrics_history['train_loss'], 'b-', label='Training Loss (log scale)', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_plot_path = os.path.join(save_dir, 'training_losses.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(loss_plot_path, "plots")
    
    def _plot_learning_rate(self, save_dir: str):
        """Plot learning rate schedule."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.metrics_history['learning_rate']) + 1)
        plt.plot(epochs, self.metrics_history['learning_rate'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        lr_plot_path = os.path.join(save_dir, 'learning_rate.png')
        plt.savefig(lr_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(lr_plot_path, "plots")
    
    def export_metrics_csv(self, filepath: str = "training_metrics.csv"):
        """Export metrics to CSV file."""
        if not self.metrics_history['train_loss']:
            return
            
        # Convert metrics history to DataFrame
        max_len = len(self.metrics_history['train_loss'])
        
        # Create DataFrame with train_loss as base
        df_data = {
            'epoch': list(range(1, max_len + 1)),
            'train_loss': self.metrics_history['train_loss'],
            'learning_rate': self.metrics_history['learning_rate'][:max_len],
            'mask_ratio': self.metrics_history['mask_ratio'][:max_len]
        }
        
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        
        # Log CSV to MLflow
        mlflow.log_artifact(filepath, "metrics")
        
        print(f"Metrics exported to {filepath}")
    
    def log_final_summary(self, best_val_loss: float, total_epochs: int):
        """Log final training summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        mlflow.log_metric("final_best_val_loss", best_val_loss)
        mlflow.log_metric("total_epochs", total_epochs)
        mlflow.log_metric("total_training_time", total_time)
        mlflow.log_metric("avg_time_per_epoch", total_time / total_epochs if total_epochs > 0 else 0)
        
        # Create summary text
        summary = f"""Training Summary:
Total Epochs: {total_epochs}
Best Validation Loss: {best_val_loss:.6f}
Total Training Time: {total_time/3600:.2f} hours
Average Time per Epoch: {total_time/total_epochs:.2f} seconds
"""
        
        # Save summary to file and log it
        summary_path = "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
        mlflow.log_artifact(summary_path, "summary")
        
        print(summary)
    
    def end_run(self):
        """End the MLflow run."""
        if mlflow.active_run():
            mlflow.end_run()
            print("MLflow run ended.")
    
    def get_run_url(self) -> str:
        """Get the MLflow UI URL for this run."""
        if mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            return f"http://localhost:5000/#/experiments/1/runs/{run_id}"
        return "No active run"


def get_gpu_memory_usage() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    return 0.0