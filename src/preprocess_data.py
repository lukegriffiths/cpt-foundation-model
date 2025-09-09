"""
This script preprocesses Cone Penetration Test (CPT) data from a raw CSV file.
The preprocessing pipeline consists of the following steps:
1. Load Data: Reads a CSV file specified in a YAML config and groups the data by CPT 'ID'.
2. Feature Engineering:
   - Dynamically processes numerical features (qc, fs, u2) and a categorical feature (soil_class)
     based on their presence in the config file.
   - Converts units (e.g., MPa to kPa for 'qc').
   - Handles missing values by filling with the median.
   - One-hot encodes the 'soil_class' categorical feature.
3. Scaling:
   - Fits a StandardScaler on the combined numerical data from all CPTs to normalize them.
   - Saves the fitted scaler object for later use during inference.
4. Tensor Conversion & Saving:
   - Applies the scaler to the numerical features.
   - Combines scaled numerical features and one-hot encoded categorical features.
   - Converts the final processed data for each CPT into a PyTorch tensor.
   - Saves each tensor as a separate .pt file in a processed data directory.

The script is driven by a YAML configuration file that specifies file paths,
feature mappings, and other parameters, allowing for flexible data processing.
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

def load_and_group_data(file_path: str) -> tuple[pd.DataFrame | None, dict | None]:
    """Loads a single CSV, groups it by 'ID', and returns the full df and the grouped dict."""
    print(f"Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None, None

    print("Grouping data by 'ID' column...")
    grouped = df.groupby('ID')
    cpt_dfs = {cpt_id: group for cpt_id, group in grouped}
    
    print(f"Found {len(cpt_dfs)} unique CPT traces.")
    return df, cpt_dfs
def preprocess_cpts(cpt_dict: dict, all_soil_classes: list, feature_mapping: dict) -> tuple[dict, list]:
    """
    Performs unit conversion, one-hot encodes soil classes, and combines features.
    Features are processed only if they are defined in the feature_mapping.
    """
    print("Preprocessing CPTs (unit conversion and one-hot encoding)...")
    processed_cpts = {}
    
    # Dynamically get original column names and clean names from the config's feature mapping
    numerical_cols_original = []
    clean_numerical_cols = []
    
    for clean_name, original_name in feature_mapping.items():
        if clean_name in ['qc', 'fs', 'u2']:
            numerical_cols_original.append(original_name)
            clean_numerical_cols.append(clean_name)

    soil_class_col = feature_mapping.get('soil_class')

    for cpt_id, df in tqdm(cpt_dict.items()):
        combined_features = []

        # --- Process Numerical Features ---
        if numerical_cols_original:
            numerical_df = df[numerical_cols_original].copy()
            
            for col in numerical_cols_original:
                numerical_df[col] = pd.to_numeric(numerical_df[col], errors='coerce')
                if numerical_df[col].isnull().any():
                    median_val = numerical_df[col].median()
                    numerical_df[col] = numerical_df[col].fillna(median_val)

            rename_dict = {orig: clean for orig, clean in zip(numerical_cols_original, clean_numerical_cols)}
            numerical_df.rename(columns=rename_dict, inplace=True)

            if 'qc' in clean_numerical_cols and '(MPa)' in feature_mapping.get('qc', ''):
                numerical_df['qc'] *= 1000
            
            combined_features.append(numerical_df.reset_index(drop=True))

        # --- Process Categorical Features ---
        if soil_class_col and soil_class_col in df.columns:
            soil_class_series = df[soil_class_col].copy().fillna('Unknown')
            soil_class_cat = pd.Categorical(soil_class_series, categories=all_soil_classes)
            one_hot_df = pd.get_dummies(soil_class_cat, prefix='soil')
            combined_features.append(one_hot_df.reset_index(drop=True))
        
        # --- Combine All Available Features ---
        if not combined_features:
            print(f"Warning: No features were processed for CPT ID {cpt_id}. Skipping.")
            continue
        
        combined_df = pd.concat(combined_features, axis=1)
        processed_cpts[cpt_id] = combined_df
        
    return processed_cpts, clean_numerical_cols

def fit_scaler_on_all_data(cpt_dict: dict, numerical_cols: list) -> StandardScaler | None:
    """Fits a StandardScaler on the combined NUMERICAL data from all CPTs."""
    if not numerical_cols:
        print("No numerical columns to scale. Skipping scaler fitting.")
        return None
        
    print(f"Fitting StandardScaler on numerical columns: {numerical_cols}...")
    combined_numerical_data = np.vstack([df[numerical_cols].values for df in cpt_dict.values()])
    
    scaler = StandardScaler().fit(combined_numerical_data)
    print("Scaler fitted successfully.")
    return scaler

def process_and_save_files(cpt_dict: dict, scaler: StandardScaler | None, numerical_cols: list, output_dir: str):
    """Scales numerical features, combines with one-hot features, and saves tensors."""
    print(f"Processing and saving {len(cpt_dict)} combined tensors to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    for cpt_id, df in tqdm(cpt_dict.items()):
        try:
            # Separate numerical and one-hot data based on the columns present
            numerical_data_list = []
            if numerical_cols:
                numerical_data = df[numerical_cols].values
                if scaler:
                    scaled_numerical = scaler.transform(numerical_data)
                    numerical_data_list.append(scaled_numerical)
                else:
                    numerical_data_list.append(numerical_data)

            one_hot_cols = [col for col in df.columns if col not in numerical_cols]
            if one_hot_cols:
                one_hot_data = df[one_hot_cols].values
                numerical_data_list.append(one_hot_data)

            if not numerical_data_list:
                print(f"Warning: No data to save for CPT ID {cpt_id}. Skipping.")
                continue

            final_features = np.hstack(numerical_data_list)
            tensor_data = torch.tensor(final_features, dtype=torch.float32)
            
            output_path = os.path.join(output_dir, f"cpt_{cpt_id}.pt")
            torch.save(tensor_data, output_path)
        except Exception as e:
            print(f"Error processing CPT with ID {cpt_id}: {e}")

def main(config: dict):
    """Main function to orchestrate the data processing pipeline using a config file."""
    paths = config['data_paths']
    feature_mapping = config['feature_mapping']

    full_df, cpt_data_dict = load_and_group_data(paths['input_file'])
    if full_df is None or cpt_data_dict is None:
        return
    
    all_soil_classes = []
    if 'soil_class' in feature_mapping and feature_mapping['soil_class'] in full_df.columns:
        all_soil_classes = full_df[feature_mapping['soil_class']].fillna('Unknown').unique().tolist()
        print(f"Found {len(all_soil_classes)} unique soil classes: {all_soil_classes}")
    else:
        print("No 'soil_class' feature defined or found. Skipping soil class processing.")

    processed_cpts_dict, numerical_cols = preprocess_cpts(cpt_data_dict, all_soil_classes, feature_mapping)
    
    if not processed_cpts_dict:
        print("No CPTs were processed successfully. Exiting.")
        return

    scaler = fit_scaler_on_all_data(processed_cpts_dict, numerical_cols)
    if scaler:
        joblib.dump(scaler, paths['scaler_path'])
        print(f"Scaler saved to '{paths['scaler_path']}'.")
    
    process_and_save_files(processed_cpts_dict, scaler, numerical_cols, paths['processed_dir'])
    print("\nData processing complete.")

if __name__ == '__main__':
    # --- Define the default configuration file to use if none is provided ---
    DEFAULT_CONFIG_PATH = 'configs/PG_dataset.yaml'

    parser = argparse.ArgumentParser(description="Preprocess CPT data using a config file.")
    
    # --- Make the --config argument optional and provide a default value ---
    parser.add_argument(
        '--config', 
        type=str, 
        default=DEFAULT_CONFIG_PATH, 
        help=f"Path to the YAML configuration file. Defaults to '{DEFAULT_CONFIG_PATH}'."
    )
    args = parser.parse_args()

    # --- Load the configuration file ---
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Using configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config}'")
        exit()
    except Exception as e:
        print(f"Error loading or parsing configuration file: {e}")
        exit()
    

    main(config)