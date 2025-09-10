import pandas as pd
import os
import yaml

def create_ic_labels(raw_data_path: str, output_path: str):
    """
    Reads raw CPT data, extracts the Soil Behavior Type Index (Ic),
    and saves it to a new CSV file for fine-tuning.

    This script assumes the raw data CSV contains the columns 'ID', 'Depth (m)',
    and 'Ic (-)'.

    Args:
        raw_data_path (str): Path to the raw CPT data CSV file.
        output_path (str): Path to save the processed labels CSV file.
    """
    print(f"Reading raw data from: {raw_data_path}")
    
    # --- 1. Load the Raw Data ---
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{raw_data_path}'")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # --- 2. Select and Rename Columns ---
    # Define the columns we need and their new, cleaner names
    required_columns = {
        'ID': 'ID',
        'Depth (m)': 'Depth',
        'Ic (-)': 'Ic'
    }

    # Check if all required columns exist in the dataframe
    if not all(col in df.columns for col in required_columns.keys()):
        print("Error: The raw data is missing one or more required columns: 'ID', 'Depth (m)', 'Ic (-)'.")
        return

    # Select and rename the columns
    labels_df = df[list(required_columns.keys())].rename(columns=required_columns)

    # --- 3. Handle Missing Values ---
    # Drop any rows where the Ic value is missing, as they cannot be used for training
    initial_rows = len(labels_df)
    labels_df.dropna(subset=['Ic'], inplace=True)
    dropped_rows = initial_rows - len(labels_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing 'Ic' values.")

    # --- 4. Save the Processed Labels ---
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the final dataframe to the specified path
    labels_df.to_csv(output_path, index=False)
    print(f"Successfully created label file with {len(labels_df)} rows.")
    print(f"Labels saved to: {output_path}")


if __name__ == '__main__':
    # Load configuration from the FINE-TUNING YAML file to get paths
    CONFIG_PATH = 'configs/PG_finetune_Ic.yaml'
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        paths = config['data_paths']
        
        # Get paths directly from the config file
        raw_data_path = paths['input_file']
        output_path = paths['labels_file']

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load or parse configuration file '{CONFIG_PATH}'. Make sure it exists and contains the correct 'data_paths' keys ('input_file', 'labels_file'). Details: {e}")
        exit()
    
    # Run the label creation process
    create_ic_labels(raw_data_path=raw_data_path, output_path=output_path)