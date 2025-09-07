import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# --- Configuration Parameters ---
INPUT_FILE = 'data/raw/Database_CPT_PremstallerGeotechnik/CPT_PremstallerGeotechnik_revised.csv'
OUTPUT_DIR = 'data/processed'
SCALER_PATH = 'data/scaler.joblib'
SOIL_CLASS_COLUMN = 'EN_ISO_14688_classes'

# Define the numerical feature columns to be used from the CSV
NUMERICAL_FEATURES = {
    'qc': 'qc (MPa)',
    'fs': 'fs (kPa)',
    'u2': 'u2 (kPa)'
}
# A consistent list of the final numerical column names
CLEAN_NUMERICAL_COLS = ['qc', 'fs', 'u2']
# ------------------------------

def load_and_group_data(file_path):
    """Loads a single CSV, groups it by 'ID', and returns the full df and the grouped dict."""
    print(f"Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None, None

    print(f"Grouping data by 'ID' column...")
    grouped = df.groupby('ID')
    cpt_dfs = {cpt_id: group for cpt_id, group in grouped}
    
    print(f"Found {len(cpt_dfs)} unique CPT traces.")
    return df, cpt_dfs

def preprocess_cpts(cpt_dict, all_soil_classes):
    """
    Performs unit conversion, one-hot encodes soil classes, and combines features.
    """
    print("Preprocessing CPTs (unit conversion and one-hot encoding)...")
    processed_cpts = {}
    
    for cpt_id, df in tqdm(cpt_dict.items()):
        # --- Process Numerical Features ---
        numerical_df = df[list(NUMERICAL_FEATURES.values())].copy()
        numerical_df.rename(columns={NUMERICAL_FEATURES['qc']: 'qc'}, inplace=True)
        numerical_df['qc'] = numerical_df['qc'] * 1000 # Convert qc from MPa to kPa
        numerical_df.rename(columns={
            NUMERICAL_FEATURES['fs']: 'fs',
            NUMERICAL_FEATURES['u2']: 'u2'
        }, inplace=True)

        # --- Process Categorical (Soil Class) Features ---
        soil_class_series = df[SOIL_CLASS_COLUMN].copy()
        soil_class_series.fillna('Unknown', inplace=True) # Handle missing classes
        
        # Convert to categorical type to ensure consistent columns after one-hot encoding
        soil_class_cat = pd.Categorical(soil_class_series, categories=all_soil_classes)
        one_hot_df = pd.get_dummies(soil_class_cat, prefix='soil')
        
        # --- Combine Numerical and Categorical Features ---
        combined_df = pd.concat([numerical_df, one_hot_df], axis=1)
        processed_cpts[cpt_id] = combined_df
        
    return processed_cpts

def fit_scaler_on_all_data(cpt_dict):
    """Fits a StandardScaler on the combined NUMERICAL data from all CPTs."""
    print("Fitting StandardScaler on all numerical data...")
    
    # Combine only the numerical data from all CPTs
    combined_numerical_data = np.vstack([df[CLEAN_NUMERICAL_COLS].values for df in cpt_dict.values()])
    
    scaler = StandardScaler()
    scaler.fit(combined_numerical_data)
    
    print("Scaler fitted successfully.")
    return scaler

def process_and_save_files(cpt_dict, scaler, output_dir):
    """
    Scales numerical features, combines with one-hot features, and saves tensors.
    """
    print(f"Processing and saving {len(cpt_dict)} combined tensors to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    for cpt_id, df in tqdm(cpt_dict.items()):
        try:
            # Separate numerical and one-hot encoded columns
            numerical_data = df[CLEAN_NUMERICAL_COLS].values
            one_hot_data = df.drop(columns=CLEAN_NUMERICAL_COLS).values
            
            # Scale ONLY the numerical data
            scaled_numerical = scaler.transform(numerical_data)
            
            # Recombine the scaled numerical data and the one-hot data
            final_features = np.hstack([scaled_numerical, one_hot_data])
            
            # Convert to a PyTorch tensor
            tensor_data = torch.tensor(final_features, dtype=torch.float32)
            
            # Save the processed tensor
            output_path = os.path.join(output_dir, f"cpt_{cpt_id}.pt")
            torch.save(tensor_data, output_path)
            
        except Exception as e:
            print(f"Error processing CPT with ID {cpt_id}: {e}")

def main():
    """Main function to orchestrate the data processing pipeline."""
    # 1. Load data and get a complete list of all unique soil classes
    full_df, cpt_data_dict = load_and_group_data(INPUT_FILE)
    if full_df is None:
        return
        
    # Get all unique soil classes from the entire dataset to ensure consistency
    all_soil_classes = full_df[SOIL_CLASS_COLUMN].fillna('Unknown').unique().tolist()
    print(f"Found {len(all_soil_classes)} unique soil classes: {all_soil_classes}")
    
    # 2. Preprocess data (unit conversions and one-hot encoding)
    processed_cpts_dict = preprocess_cpts(cpt_data_dict, all_soil_classes)
    
    # 3. Fit the scaler on the numerical part of the dataset
    scaler = fit_scaler_on_all_data(processed_cpts_dict)
    
    # 4. Save the fitted scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to '{SCALER_PATH}'.")
    
    # 5. Scale numerical data, combine with one-hot, and save final tensors
    process_and_save_files(processed_cpts_dict, scaler, OUTPUT_DIR)
    
    print("\nData processing complete.")

if __name__ == '__main__':
    main()