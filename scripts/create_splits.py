import pandas as pd
import numpy as np
import yaml
import argparse
import os

def create_splits(config_path: str):
    """
    Reads the raw data file, gets all unique CPT IDs, shuffles them,
    and saves them into train, validation, and test split files based on
    ratios defined in the config file.
    """
    # Load the config to get paths and split ratios
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    input_file = config['data_paths']['input_file']
    
    # --- UPDATED: Load split ratios and random seed from the config file ---
    split_config = config['data_split']
    train_ratio = split_config['train_ratio']
    val_ratio = split_config['val_ratio']
    random_seed = split_config.get('random_seed') # .get() is safer

    # Sanity check to ensure ratios are valid
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
    assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1"
    assert (train_ratio + val_ratio) < 1, "The sum of train_ratio and val_ratio must be less than 1"

    print(f"Loading IDs from {input_file}...")
    df = pd.read_csv(input_file)
    all_ids = df['ID'].unique()

    # --- UPDATED: Use the random seed for shuffling ---
    if random_seed is not None:
        print(f"Using random seed: {random_seed}")
        np.random.seed(random_seed)
        
    np.random.shuffle(all_ids) # Shuffle IDs randomly

    # Calculate split indices based on the loaded ratios
    train_end = int(len(all_ids) * train_ratio)
    val_end = train_end + int(len(all_ids) * val_ratio)

    # Create the splits
    train_ids = all_ids[:train_end]
    val_ids = all_ids[train_end:val_end]
    test_ids = all_ids[val_end:]

    print(f"Total IDs: {len(all_ids)}")
    print(f"Training set size: {len(train_ids)} ({train_ratio:.0%})")
    print(f"Validation set size: {len(val_ids)} ({val_ratio:.0%})")
    print(f"Test set size: {len(test_ids)} (~{1 - train_ratio - val_ratio:.0%})")

    # Define output directory and save the files
    output_dir = 'data/splits'
    os.makedirs(output_dir, exist_ok=True)
    
    np.savetxt(os.path.join(output_dir, 'train_ids.txt'), train_ids, fmt='%d')
    np.savetxt(os.path.join(output_dir, 'val_ids.txt'), val_ids, fmt='%d')
    np.savetxt(os.path.join(output_dir, 'test_ids.txt'), test_ids, fmt='%d')

    print(f"Split ID files saved to '{output_dir}'.")

if __name__ == '__main__':
    DEFAULT_CONFIG_PATH = 'configs/PG_dataset.yaml'
    parser = argparse.ArgumentParser(description="Create train/val/test splits for CPT data.")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help="Path to the main YAML config file.")
    args = parser.parse_args()
    create_splits(args.config)