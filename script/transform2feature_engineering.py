import pandas as pd
import yaml
import argparse
from datetime import datetime
import os


parser = argparse.ArgumentParser(description="Read data from a CSV file using configuration file.")
parser.add_argument('--config', type=str, default='../config.yaml', help="Path to the configuration YAML file.")


def read_csv(config_path: str = parser.parse_args().config) -> pd.DataFrame:
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

    data_path = os.listdir(os.path.join(config['WORK_DIR'], config['INGESTION_PATH']))[-1]
    df = pd.read_csv(data_path)

    return df

def transform(config_path: str = parser.parse_args().config) -> pd.DataFrame:
    ## ทำต่อ..

if __name__ == '__main__':
    df = read_csv()
    df = df[~df['Name'].isna()]

    df['ID'] = df['ID'].astype('object')
    df['Brand'] = df['Brand'].astype('category')
    df['Colour'] = df['Colour'].astype('category')
    sub_categories = df['Catagories'].str.split(',', expand=True)
    sub_categories = sub_categories.fillna('no-sub-categories')
    sub_categories.columns = [f'sub-Catagories-{i+1}' for i in range(sub_categories.shape[1])]

    df = pd.concat([df, sub_categories], axis=1)
    df = df.drop(columns=['Catagories'])
    df = df[df['Price'] < 400]