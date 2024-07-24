import pandas as pd
import yaml
import argparse

parser = argparse.ArgumentParser(description="Read data from a CSV file using configuration file.")
parser.add_argument('--config', type=str, default='../config.yaml', help="Path to the configuration YAML file.")

def read_data(config_path: str = parser.parse_args().config) -> pd.DataFrame:
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
    return df

# if __name__ == '__main__':
#     args = parser.parse_args()
    
#     df = read_data(args.config)
