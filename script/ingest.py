import pandas as pd
import yaml
import argparse
from datetime import datetime
import os
datetimerun = datetime.now().strftime("%Y-%m-%d-%H-%M")

parser = argparse.ArgumentParser(description="Read data from a CSV file using configuration file.")
parser.add_argument('--config', type=str, default='../config.yaml', help="Path to the configuration YAML file.")

def save_ingestion(df, config):
    """
    Save the ingestion DataFrame to a specified directory structure.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - config (dict): Configuration dictionary containing 'WORK_DIR' and 'INGESTION_PATH'.
    
    Returns:
    - None
    """
    datetimerun = datetime.now().strftime("%Y-%m-%d-%H-%M")
    ingest_saved_path = os.path.join(config['WORK_DIR'], config['INGESTION_PATH'], datetimerun)
    
    # Create the ingestion path if it does not exist
    if not os.path.exists(os.path.join(config['WORK_DIR'], config['INGESTION_PATH'])):
        os.makedirs(os.path.join(config['WORK_DIR'], config['INGESTION_PATH']))
        
    # Save the DataFrame if the ingest_saved_path does not exist
    if not os.path.exists(ingest_saved_path):
        os.makedirs(ingest_saved_path)
        df.to_csv(os.path.join(ingest_saved_path, 'ingest.csv'), index=False)
        print('##### Ingestion is saved to {}'.format(os.path.join(ingest_saved_path, 'ingest.csv')))
    else:
        print(f'{ingest_saved_path} already exists!')

def read2save_data(config_path: str = parser.parse_args().config) -> pd.DataFrame:
    """
    Reads data from a CSV file specified in the configuration file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the CSV file.
    
    Example:
        df = read_data('../config.yaml')
        print(df.head())
        
        Output:
           dress_id  price
        0         1   20.0
        1         2   35.0
        2         3   15.0
        3         4   25.0
        4         5   30.0
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    data_path = config['DATA_PATH']
    df = pd.read_csv(data_path)
    save_ingestion(df, config)

if __name__ == '__main__':
    args = parser.parse_args()
    read2save_data(args.config)
    
