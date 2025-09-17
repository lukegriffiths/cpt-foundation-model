# MLflow Integration for CPT Foundation Model

This directory contains MLflow integration for comprehensive experiment tracking, metrics logging, and model management for the CPT Foundation Model training.

## Quick Start

1. **Update your conda environment:**
   ```bash
   conda env update -f environment.yml
   # OR if you need to recreate:
   conda env remove -n cpt-fm
   conda env create -f environment.yml
   conda activate cpt-fm
   ```

2. **Verify MLflow setup (optional):**
   ```bash
   python scripts/setup_mlflow.py
   ```

3. **Run training with MLflow tracking:**
   ```bash
   python src/train.py --config configs/PG_dataset.yaml
   ```

4. **View results in MLflow UI:**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   # OR
   python scripts/setup_mlflow.py --ui
   ```
   Then open: http://localhost:5000

## Features

### Automatic Tracking
- **Metrics**: Training/validation loss, learning rate, mask ratio, GPU memory usage
- **Parameters**: All configuration parameters from YAML files
- **Models**: Best model checkpoints and regular checkpoints
- **Artifacts**: Training plots, metrics CSV files, model files

### Visualizations
- **Training loss curves** (linear and log scale)
- **Learning rate schedules**
- **GPU memory usage**
- **Training dashboard** with all key metrics

### Experiment Analysis
```bash
# Compare all runs in an experiment
python scripts/analyze_experiments.py --experiment cpt-foundation-training

# Detailed analysis of a specific run
python scripts/analyze_experiments.py --run-id <run_id>

# Custom output directory
python scripts/analyze_experiments.py --output-dir my_analysis
```

## Directory Structure

After running training with MLflow:

```
├── mlruns/                     # MLflow tracking data
├── plots/                      # Training plots
├── training_metrics.csv        # Exported metrics
├── analysis_plots/             # Experiment analysis plots
└── models/                     # Model checkpoints
```

## MLflow UI Features

1. **Experiments View**: Compare multiple training runs
2. **Run Details**: Detailed metrics, parameters, and artifacts for each run
3. **Model Registry**: Track and version your best models
4. **Metrics Comparison**: Side-by-side comparison of runs
5. **Artifact Browser**: Download plots, models, and other files

## Key Metrics Tracked

- `train_loss`: Training loss per epoch
- `val_loss`: Validation loss (every 5 epochs)
- `learning_rate`: Current learning rate
- `mask_ratio`: Actual masking ratio used
- `gpu_memory_mb`: GPU memory usage in MB
- `epoch_time`: Time per epoch
- `final_best_val_loss`: Best validation loss achieved
- `total_training_time`: Total training duration
- `total_epochs`: Number of epochs completed

## Model Management

MLflow automatically tracks:
- **Best models**: Saved when validation loss improves
- **Regular checkpoints**: Saved every 10 epochs
- **Model registry**: Best models are registered for easy deployment

## Advanced Usage

### Custom Experiment Names
Add to your config file:
```yaml
training_params:
  experiment_name: "my-custom-experiment"
  run_name: "experiment-v1"
```

### Remote MLflow Tracking
```python
# In your config or code
mlflow_tracking_uri: "http://your-mlflow-server:5000"
```

### Export and Analysis
```bash
# Export metrics to CSV
# (automatically done during training)

# Create custom analysis plots
python scripts/analyze_experiments.py

# Access MLflow programmatically
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
experiments = client.list_experiments()
print([exp.name for exp in experiments])
"
```

## Troubleshooting

1. **MLflow UI not starting**: Make sure port 5000 is available
2. **Import errors**: Install dependencies with `pip install -r requirements-mlflow.txt`
3. **No experiments showing**: Check that training was run with MLflow tracking enabled
4. **Large artifact files**: MLflow stores all plots and model files - monitor disk space

## Integration with Other Tools

MLflow integrates well with:
- **Jupyter Notebooks**: Load and analyze experiments
- **TensorBoard**: Can be used alongside MLflow
- **Model deployment**: Direct deployment from MLflow model registry
- **CI/CD**: Automated model validation and deployment