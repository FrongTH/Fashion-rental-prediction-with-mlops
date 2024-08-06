import mlflow
from sklearn.metrics import mean_squared_error, root_mean_squared_error
# from sklearn.linear_model import Lasso, BayesianRidge, SGDRegressor
from tqdm import tqdm
from hpsklearn import HyperoptEstimator, any_regressor
from hyperopt import tpe
from glob import glob
import os
from mlflow import MlflowClient

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import yaml
import argparse

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType




mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fashion-rental-prediction")
# Initialize the MLflow client
client = MlflowClient()
parser = argparse.ArgumentParser(description="Read data from a CSV file using configuration file.")
parser.add_argument('--config', type=str, default='../config.yaml', help="Path to the configuration YAML file.")


def read_csv(path: str) -> pd.DataFrame:
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
    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    # lasso = Lasso(alpha=0.1)
    # bayesian = BayesianRidge()
    # sgd = SGDRegressor()

    estim = HyperoptEstimator(regressor=any_regressor('reg'),
                                algo=tpe.suggest, 
                                trial_timeout=300,
                          )
    args = parser.parse_args()
    
    with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
    lastest_path = glob(os.path.join(config['WORK_DIR'], config['DATA_ROOT'], config['SPLIT_PATH'], '*'))[-1]

    x_train = read_csv(os.path.join(lastest_path, config['TRAIN_PATH'],'x_train.csv'))
    y_train = read_csv(os.path.join(lastest_path, config['TRAIN_PATH'],'y_train.csv'))
        
    x_validation = read_csv(os.path.join(lastest_path, config['VALIDATION_PATH'],'x_validation.csv'))
    y_validation = read_csv(os.path.join(lastest_path, config['VALIDATION_PATH'],'y_validation.csv'))
    
    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:
        estim.fit(x_train, y_train)

    # Get the experiment by name
    experiment_name = "fashion-rental-prediction"
    experiment = client.get_experiment_by_name(experiment_name)

    best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type = ViewType.ACTIVE_ONLY,
            max_results=5,
            order_by=["metrics.training_root_mean_squared_error"]
        )[0]

    run_id = best_run.info.run_id
    model_uri = "runs/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="best_optimized_model")
