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

def preprocess_cpts(cpt_dict: dict, all_soil_classes: list, feature_mapping: dict) -> dict:
    """Performs unit conversion, one-hot encodes soil classes, and combines features."""
    print("Preprocessing CPTs (unit conversion and one-hot encoding)...")
    processed_cpts = {}
    
    # Get original column names from the config's feature mapping
    qc_col = feature_mapping['qc']
    fs_col = feature_mapping['fs']
    u2_col = feature_mapping['u2']
    soil_class_col = feature_mapping['soil_class']
    
    numerical_cols_original = [qc_col, fs_col, u2_col]
    
    for cpt_id, df in tqdm(cpt_dict.items()):
        # --- Process Numerical Features ---
        numerical_df = df[numerical_cols_original].copy()
        numerical_df.rename(columns={qc_col: 'qc', fs_col: 'fs', u2_col: 'u2'}, inplace=True)
        
        # Specific transformation: Convert qc from MPa to kPa
        if '(MPa)' in qc_col:
            numerical_df['qc'] = numerical_df['qc'] * 1000
        
        # --- Process Categorical Features ---
        soil_class_series = df[soil_class_col].copy().fillna('Unknown')
        soil_class_cat = pd.Categorical(soil_class_series, categories=all_soil_classes)
        one_hot_df = pd.get_dummies(soil_class_cat, prefix='soil')
        
        # --- Combine Features ---
        combined_df = pd.concat([numerical_df, one_hot_df.reset_index(drop=True)], axis=1)
        processed_cpts[cpt_id] = combined_df
        
    return processed_cpts

def fit_scaler_on_all_data(cpt_dict: dict) -> StandardScaler:
    """Fits a StandardScaler on the combined NUMERICAL data from all CPTs."""
    print("Fitting StandardScaler on all numerical data...")
    clean_numerical_cols = ['qc', 'fs', 'u2']
    combined_numerical_data = np.vstack([df[clean_numerical_cols].values for df in cpt_dict.values()])
    
    scaler = StandardScaler().fit(combined_numerical_data)
    print("Scaler fitted successfully.")
    return scaler

def process_and_save_files(cpt_dict: dict, scaler: StandardScaler, output_dir: str):
    """Scales numerical features, combines with one-hot features, and saves tensors."""
    print(f"Processing and saving {len(cpt_dict)} combined tensors to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    clean_numerical_cols = ['qc', 'fs', 'u2']
    
    for cpt_id, df in tqdm(cpt_dict.items()):
        try:
            numerical_data = df[clean_numerical_cols].values
            one_hot_data = df.drop(columns=clean_numerical_cols).values
            
            scaled_numerical = scaler.transform(numerical_data)
            final_features = np.hstack([scaled_numerical, one_hot_data])
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
    if full_df is None:
        return
        
    all_soil_classes = full_df[feature_mapping['soil_class']].fillna('Unknown').unique().tolist()
    print(f"Found {len(all_soil_classes)} unique soil classes: {all_soil_classes}")
    
    processed_cpts_dict = preprocess_cpts(cpt_data_dict, all_soil_classes, feature_mapping)
    
    scaler = fit_scaler_on_all_data(processed_cpts_dict)
    joblib.dump(scaler, paths['scaler_path'])
    print(f"Scaler saved to '{paths['scaler_path']}'.")
    
    process_and_save_files(processed_cpts_dict, scaler, paths['processed_dir'])
    print("\nData processing complete.")

if __name__ == '__main__':
    # example usage: python src/preprocess_data.py --config configs/PG_dataset.yaml
    parser = argparse.ArgumentParser(description="Preprocess CPT data using a config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)