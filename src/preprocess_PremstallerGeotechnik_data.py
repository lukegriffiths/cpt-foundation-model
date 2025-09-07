import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib

# --- Configuration Parameters ---
INPUT_FILE = 'data/raw/your_single_cpt_file.csv'  # <-- IMPORTANT: Set the name of your CSV file here
OUTPUT_DIR = 'data/processed'
SCALER_PATH = 'data/scaler.joblib'

# Define the feature columns to be used from the CSV
# We will convert qc from MPa to kPa to be consistent with fs and u2
ORIGINAL_FEATURES = {
    'qc': 'qc (MPa)',
    'fs': 'fs (kPa)',
    'u2': 'u2 (kPa)'
}
# ------------------------------

def load_and_group_data(file_path):
    """Loads a single CSV and groups it by the 'ID' column."""
    print(f"Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None, None

    # Group data into a dictionary of DataFrames, with IDs as keys
    print(f"Grouping data by 'ID' column...")
    grouped = df.groupby('ID')
    cpt_dfs = {cpt_id: group for cpt_id, group in grouped}
    
    print(f"Found {len(cpt_dfs)} unique CPT traces.")
    return cpt_dfs

def preprocess_cpts(cpt_dict):
    """
    Performs unit conversion (qc MPa -> kPa) and selects feature columns.
    """
    processed_cpts = {}
    feature_cols = list(ORIGINAL_FEATURES.values())
    
    for cpt_id, df in cpt_dict.items():
        # Create a copy to avoid SettingWithCopyWarning
        processed_df = df[feature_cols].copy()
        
        # --- Unit Conversion ---
        # Convert qc from MPa to kPa
        processed_df.rename(columns={ORIGINAL_FEATURES['qc']: 'qc'}, inplace=True)
        processed_df['qc'] = processed_df['qc'] * 1000
        
        # Rename other columns for consistency
        processed_df.rename(columns={
            ORIGINAL_FEATURES['fs']: 'fs',
            ORIGINAL_FEATURES['u2']: 'u2'
        }, inplace=True)
        
        processed_cpts[cpt_id] = processed_df
        
    return processed_cpts


def fit_scaler_on_all_data(cpt_dict):
    """Fits a StandardScaler on the combined data from all CPTs."""
    print("Fitting StandardScaler on all data...")
    
    # Combine data from all CPTs into a single NumPy array
    combined_data = np.vstack([df.values for df in cpt_dict.values()])
    
    scaler = StandardScaler()
    scaler.fit(combined_data)
    
    print("Scaler fitted successfully.")
    return scaler

def process_and_save_files(cpt_dict, scaler, output_dir):
    """
    Scales each CPT trace and saves it as a PyTorch tensor.
    """
    print(f"Processing and saving {len(cpt_dict)} files to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    
    for cpt_id, df in tqdm(cpt_dict.items()):
        try:
            # Scale the data using the fitted scaler
            scaled_features = scaler.transform(df.values)
            
            # Convert to a PyTorch tensor
            tensor_data = torch.tensor(scaled_features, dtype=torch.float32)
            
            # Save the processed tensor using its ID in the filename
            output_path = os.path.join(output_dir, f"cpt_{cpt_id}.pt")
            torch.save(tensor_data, output_path)
            
        except Exception as e:
            print(f"Error processing CPT with ID {cpt_id}: {e}")

def main():
    """Main function to orchestrate the data processing pipeline."""
    # 1. Load and group the data from the single CSV
    cpt_data_dict = load_and_group_data(INPUT_FILE)
    if not cpt_data_dict:
        return
        
    # 2. Preprocess data (unit conversions, column selection)
    processed_cpts_dict = preprocess_cpts(cpt_data_dict)
    
    # 3. Fit the scaler on the entire dataset
    scaler = fit_scaler_on_all_data(processed_cpts_dict)
    
    # 4. Save the fitted scaler for future use
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to '{SCALER_PATH}'.")
    
    # 5. Process each CPT group and save the output tensors
    process_and_save_files(processed_cpts_dict, scaler, OUTPUT_DIR)
    
    print("\nData processing complete.")

if __name__ == '__main__':
    main()