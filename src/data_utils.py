"""
This script provides a CPTDataModule class to handle all data-related tasks,
from preprocessing raw CSV files to creating a PyTorch DataLoader.

The pipeline is driven by a YAML configuration file. The CPTDataModule can be
instantiated in the main training script to provide a clean interface to the data.

Key functionalities:
- Preprocessing raw data (if processed data does not exist).
- Creating a PyTorch Dataset from the processed tensor files.
- Providing a DataLoader for model training.
"""

import os
import yaml
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class CPTTensorDataset(Dataset):
    """
    Loads pre-processed CPT tensor files, chunks long sequences, 
    and applies to a specific subset of CPT IDs.
    """
    def __init__(self, processed_dir: str, cpt_ids: list, max_len: int, overlap: int):
        self.chunks = []
        
        # Get all file paths for the given CPT IDs
        file_paths = [
            os.path.join(processed_dir, f"cpt_{cpt_id}.pt") 
            for cpt_id in cpt_ids
        ]
        # Filter out paths that might not exist
        file_paths = [p for p in file_paths if os.path.exists(p)]

        print(f"Processing {len(file_paths)} files with max_len={max_len} and overlap={overlap}...")
        for path in tqdm(file_paths, desc="Loading and Chunking Data"):
            tensor = torch.load(path)
            
            # If the tensor is shorter than or equal to max_len, just add it
            if tensor.shape[0] <= max_len:
                self.chunks.append(tensor)
            # Otherwise, create overlapping chunks
            else:
                start = 0
                while start < tensor.shape[0]:
                    end = start + max_len
                    chunk = tensor[start:end]
                    
                    # If the last chunk is too small, it might be better to ignore it
                    # or handle it differently, but for now, we include it.
                    self.chunks.append(chunk)
                    
                    # Move the window for the next chunk
                    if end >= tensor.shape[0]:
                        break
                    start += max_len - overlap

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.chunks[idx]


def collate_cpts(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """ Custom collate function
    Pads sequences in a batch and creates an attention mask."""
    padding_value = -9999.0
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=padding_value)
    lengths = [len(seq) for seq in batch]
    attention_mask = torch.zeros(padded_batch.shape[:2], dtype=torch.float32)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1.0
    return padded_batch, attention_mask


class CPTDataModule:
    """Main datamodule class.
    A class to handle the entire data pipeline from raw CSV to DataLoader."""
    def __init__(self, config: dict):
        self.config = config
        self.paths = config['data_paths']
        self.feature_mapping = config['feature_mapping']
        
        # Datasets for each stage
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """
        Runs preprocessing if needed and sets up train, val, and test datasets.
        """
        if not os.path.exists(self.paths['processed_dir']) or not os.listdir(self.paths['processed_dir']):
            print("Processed data not found. Running preprocessing...")
            os.makedirs(self.paths['processed_dir'], exist_ok=True)
            self._preprocess()
        else:
            print(f"Found existing processed data in '{self.paths['processed_dir']}'. Delete to reprocess.")

        # Load the split IDs
        splits_dir = 'data/splits'
        train_ids = np.loadtxt(os.path.join(splits_dir, 'train_ids.txt'), dtype=int).tolist()
        val_ids = np.loadtxt(os.path.join(splits_dir, 'val_ids.txt'), dtype=int).tolist()
        test_ids = np.loadtxt(os.path.join(splits_dir, 'test_ids.txt'), dtype=int).tolist()

        # Create datasets for each split
        max_len = self.config['training_params'].get('max_len', 2048)
        overlap = self.config['training_params'].get('overlap', 256)
        
        self.train_dataset = CPTTensorDataset(self.paths['processed_dir'], train_ids, max_len, overlap)
        self.val_dataset = CPTTensorDataset(self.paths['processed_dir'], val_ids, max_len, overlap)
        self.test_dataset = CPTTensorDataset(self.paths['processed_dir'], test_ids, max_len, overlap)
        
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def get_dataloader(self, stage: str, shuffle: bool = True) -> DataLoader:
        """Returns a DataLoader for the specified stage ('train', 'val', or 'test')."""
        if stage == 'train':
            dataset = self.train_dataset
        elif stage == 'val':
            dataset = self.val_dataset
        elif stage == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError(f"Stage '{stage}' not recognized. Use 'train', 'val', or 'test'.")

        if dataset is None:
             raise RuntimeError("Dataset not set up. Please run data_module.setup() first.")

        batch_size = self.config['training_params']['batch_size']
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_cpts, num_workers=4, pin_memory=True)

    def _preprocess(self):
        """The main preprocessing pipeline, encapsulated as a private method."""
        full_df, cpt_data_dict = self._load_and_group_data(self.paths['input_file'])
        if full_df is None: return

        all_soil_classes = []
        if 'soil_class' in self.feature_mapping and self.feature_mapping['soil_class'] in full_df.columns:
            all_soil_classes = full_df[self.feature_mapping['soil_class']].fillna('Unknown').unique().tolist()
        
        processed_cpts_dict, numerical_cols = self._preprocess_cpts(cpt_data_dict, all_soil_classes)
        if not processed_cpts_dict: return

        scaler = self._fit_scaler(processed_cpts_dict, numerical_cols)
        if scaler:
            joblib.dump(scaler, self.paths['scaler_path'])
        
        self._process_and_save_tensors(processed_cpts_dict, scaler, numerical_cols, self.paths['processed_dir'])
        print("\nPreprocessing complete.")

    def _load_and_group_data(self, file_path: str):
        # (Logic from your original load_and_group_data function)
        print(f"Loading data from '{file_path}'...")
        try:
            df = pd.read_csv(file_path)
            grouped = df.groupby('ID')
            cpt_dfs = {cpt_id: group for cpt_id, group in grouped}
            print(f"Found {len(cpt_dfs)} unique CPT traces.")
            return df, cpt_dfs
        except FileNotFoundError:
            print(f"Error: Input file not found at '{file_path}'")
            return None, None
    
    def _preprocess_cpts(self, cpt_dict: dict, all_soil_classes: list):
        # (Logic from your original preprocess_cpts function)
        processed_cpts, clean_numerical_cols = {}, []
        numerical_cols_original = []
        
        for clean, original in self.feature_mapping.items():
            if clean in ['qc', 'fs', 'u2']:
                numerical_cols_original.append(original)
                clean_numerical_cols.append(clean)
        soil_class_col = self.feature_mapping.get('soil_class')

        for cpt_id, df in tqdm(cpt_dict.items(), desc="Preprocessing CPTs"):
            combined_features = []
            if numerical_cols_original:
                numerical_df = df[numerical_cols_original].copy()
                for col in numerical_cols_original:
                    numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce').fillna(numerical_df[col].median())
                rename_dict = {orig: clean for orig, clean in zip(numerical_cols_original, clean_numerical_cols)}
                numerical_df.rename(columns=rename_dict, inplace=True)
                if 'qc' in clean_numerical_cols and '(MPa)' in self.feature_mapping.get('qc', ''):
                    numerical_df['qc'] *= 1000
                combined_features.append(numerical_df.reset_index(drop=True))

            if soil_class_col and soil_class_col in df.columns:
                soil_series = pd.Categorical(df[soil_class_col].fillna('Unknown'), categories=all_soil_classes)
                one_hot_df = pd.get_dummies(soil_series, prefix='soil')
                combined_features.append(one_hot_df.reset_index(drop=True))

            if combined_features:
                processed_cpts[cpt_id] = pd.concat(combined_features, axis=1)

        return processed_cpts, clean_numerical_cols
    
    def _fit_scaler(self, cpt_dict: dict, numerical_cols: list):
        # (Logic from your original fit_scaler_on_all_data function)
        if not numerical_cols: return None
        print(f"Fitting StandardScaler on: {numerical_cols}...")
        combined_data = np.vstack([df[numerical_cols].values for df in cpt_dict.values()])
        return StandardScaler().fit(combined_data)

    def _process_and_save_tensors(self, cpt_dict: dict, scaler: StandardScaler | None, numerical_cols: list, output_dir: str):
        # (Logic from your original process_and_save_files function)
        print(f"Saving {len(cpt_dict)} tensors to '{output_dir}'...")
        os.makedirs(output_dir, exist_ok=True)
        for cpt_id, df in tqdm(cpt_dict.items(), desc="Saving Tensors"):
            # ... (your original saving logic) ...
            numerical_data_list, one_hot_cols = [], [col for col in df.columns if col not in numerical_cols]
            if numerical_cols:
                data = df[numerical_cols].values
                if scaler: data = scaler.transform(data)
                numerical_data_list.append(data)
            if one_hot_cols: numerical_data_list.append(df[one_hot_cols].values)
            if not numerical_data_list: continue
            
            final_features = np.hstack(numerical_data_list)
            tensor_data = torch.tensor(final_features, dtype=torch.float32)
            torch.save(tensor_data, os.path.join(output_dir, f"cpt_{cpt_id}.pt"))


if __name__ == '__main__':
    """This block allows you to run preprocessing directly from the command line."""
    DEFAULT_CONFIG_PATH = 'configs/PG_dataset.yaml'
    parser = argparse.ArgumentParser(description="Preprocess CPT data using a config file.")
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help=f"Path to the YAML config file. Defaults to '{DEFAULT_CONFIG_PATH}'.")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Using configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Config file not found at '{args.config}'")
        exit()

    # Instantiate the module and run preprocessing
    data_module = CPTDataModule(config)
    data_module.setup()