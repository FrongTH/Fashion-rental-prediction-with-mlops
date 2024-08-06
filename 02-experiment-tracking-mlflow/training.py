import pandas as pd
import yaml
import argparse
from datetime import datetime
import os
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction import DictVectorizer
import warnings
from mlflow.tracking import MlflowClient
import mlflow
from hpsklearn import HyperoptEstimator, any_regressor
from mlflow.entities import ViewType
from hyperopt import tpe
import pickle

warnings.filterwarnings('ignore')

datetimerun = datetime.now().strftime("%Y-%m-%d-%H-%M")


parser = argparse.ArgumentParser(description="Read data from a CSV file using configuration file.")
parser.add_argument('--config', type=str, default='../config.yaml', help="Path to the configuration YAML file.")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fashion-rental-prediction")
# Initialize the MLflow client
client = MlflowClient()
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

    data_path = glob(f'{os.path.join(config["WORK_DIR"], config["DATA_ROOT"], config["INGESTION_PATH"])}/*/*ingest.csv')[-1]
    df = pd.read_csv(data_path)
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the given DataFrame by performing the following steps:
    
    1. Removes rows where the 'Name' column is NaN.
    2. Converts the 'ID' column to type 'object'.
    3. Converts the 'Brand' and 'Colour' columns to type 'category'.
    4. Splits the 'Catagories' column into multiple sub-categories.
    5. Fills missing sub-categories with 'no-sub-categories'.
    6. Renames the new sub-category columns.
    7. Concatenates the new sub-category columns to the original DataFrame.
    8. Drops the original 'Catagories' column.
    9. Filters the DataFrame to only include rows where the 'Price' is less than 400.

    Args:
        df (pd.DataFrame): The input DataFrame to transform.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    
    Example:
        >>> data = {
        >>>     'Name': ['Product1', None, 'Product3'],
        >>>     'ID': [1, 2, 3],
        >>>     'Brand': ['BrandA', 'BrandB', 'BrandC'],
        >>>     'Colour': ['Red', 'Blue', 'Green'],
        >>>     'Catagories': ['Cat1,Cat2', 'Cat1', 'Cat3,Cat4'],
        >>>     'Price': [350, 450, 300]
        >>> }
        >>> df = pd.DataFrame(data)
        >>> transformed_df = transform(df)
        >>> print(transformed_df)
    """
    
    # Remove rows where 'Name' is NaN
    df = df[~df['Name'].isna()]

    # Convert 'ID' to type 'object'
    df['ID'] = df['ID'].astype('object')

    # Convert 'Brand' and 'Colour' to type 'category'
    df['Brand'] = df['Brand'].astype('category')
    df['Colour'] = df['Colour'].astype('category')

    # Split 'Catagories' into multiple sub-categories
    sub_categories = df['Catagories'].str.split(',', expand=True)
    sub_categories = sub_categories.fillna('no-sub-categories')
    sub_categories.columns = [f'sub-Catagories-{i+1}' for i in range(sub_categories.shape[1])]

    # Concatenate the new sub-category columns to the original DataFrame
    df = pd.concat([df, sub_categories], axis=1)

    # Drop the original 'Catagories' column
    df = df.drop(columns=['Catagories'])

    # Filter rows where 'Price' is less than 400
    df = df[df['Price'] < 400]

    return df, sub_categories

def feature_engineering(df: pd.DataFrame, sub_cat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the given DataFrame by adding new columns:
    
    1. 'Avg_Price_By_Brand': The average price of products grouped by 'Brand'.
    2. 'Avg_Price_By_Brand_Colour': The average price of products grouped by both 'Brand' and 'Colour'.
    3. 'Category_Count': The count of sub-categories for each product.

    Args:
        df (pd.DataFrame): The input DataFrame containing product data.
        sub_cat_df (pd.DataFrame): The DataFrame containing sub-category data for each product.

    Returns:
        pd.DataFrame: The DataFrame with new features added.
    
    Example:
        >>> data = {
        >>>     'Name': ['Product1', 'Product2', 'Product3'],
        >>>     'ID': [1, 2, 3],
        >>>     'Brand': ['BrandA', 'BrandA', 'BrandB'],
        >>>     'Colour': ['Red', 'Blue', 'Red'],
        >>>     'Price': [100, 200, 150]
        >>> }
        >>> sub_cat_data = {
        >>>     'sub-Catagories-1': ['Cat1', 'Cat1', 'Cat2'],
        >>>     'sub-Catagories-2': ['Cat2', 'no-sub-categories', 'Cat3']
        >>> }
        >>> df = pd.DataFrame(data)
        >>> sub_cat_df = pd.DataFrame(sub_cat_data)
        >>> engineered_df = feature_engineering(df, sub_cat_df)
        >>> print(engineered_df)
    """
    
    # Calculate the average price of products grouped by 'Brand'
    df['Avg_Price_By_Brand'] = df.groupby('Brand')['Price'].transform('mean')
    
    # Calculate the average price of products grouped by both 'Brand' and 'Colour'
    df['Avg_Price_By_Brand_Colour'] = df.groupby(['Brand', 'Colour'])['Price'].transform('mean')
    
    # Calculate the count of sub-categories for each product
    df['Category_Count'] = (sub_cat_df != 'no-sub-categories').sum(axis=1)
    
    return df

def split_dataframe(data, target_column, train_size=0.7, validation_size=0.2, test_size=0.1, stratify=None):
    """
    Splits a DataFrame into train, validation, and test sets, and returns features and target as NumPy arrays.
    
    Parameters:
    data (DataFrame): The input DataFrame to be split.
    target_column (str): The name of the target column.
    train_size (float): Proportion of the dataset to include in the train split (0 to 1).
    validation_size (float): Proportion of the dataset to include in the validation split (0 to 1).
    test_size (float): Proportion of the dataset to include in the test split (0 to 1).
    stratify (str or None): Column to be used for stratification. Default is None.
    
    Returns:
    x_train (ndarray): Training set features.
    y_train (ndarray): Training set target.
    x_validation (ndarray): Validation set features.
    y_validation (ndarray): Validation set target.
    x_test (ndarray): Test set features.
    y_test (ndarray): Test set target.

    Example:
    >>> import pandas as pd
    >>> data = {'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'Feature1': [0.1, 0.2, 0.2, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'Feature2': [1.1, 1.2, 1.2, 1.4, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
                'Target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    >>> df = pd.DataFrame(data)
    >>> x_train, y_train, x_validation, y_validation, x_test, y_test = split_dataframe(df, target_column='Target', stratify='Target')
    >>> print(x_train.shape, y_train.shape)
    >>> print(x_validation.shape, y_validation.shape)
    >>> print(x_test.shape, y_test.shape)
    """
    
    # Stratify parameter for the split (can be None)
    stratify_param = None
    if stratify is not None:
        stratify_param = data[stratify]
    
    # Convert DataFrame to a dictionary of records
    data_dict = data.drop(columns=[target_column, 'ID', 'Name']).to_dict(orient="records")
    
    # Vectorize the dictionary of records
    vec = DictVectorizer(sparse=False)
    data_features = vec.fit_transform(data_dict)
    
    # Split the feature matrix and target array into train+validation and test sets
    train_validation_features, test_features, train_validation_target, test_target = train_test_split(
        data_features, data[target_column].values, test_size=test_size, random_state=42, stratify=stratify_param
    )
    
    if validation_size == 0:
        x_train = train_validation_features
        y_train = train_validation_target
        x_validation, y_validation = None, None
    else:
        # Adjust validation size to account for the test set already being removed
        adjusted_validation_size = validation_size / (1 - test_size)
        
        # Split the remaining data into train and validation sets
        x_train, x_validation, y_train, y_validation = train_test_split(
            train_validation_features, train_validation_target, test_size=adjusted_validation_size, random_state=42, stratify=stratify_param
        )
    
    x_test = test_features
    y_test = test_target
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test, vec

def save_data(x_train, y_train, x_validation, y_validation, x_test, y_test, config_path: str):
    """
    Save the split data into train, validation, and test directories as CSV files.
    
    Parameters:
    - x_train, y_train, x_validation, y_validation, x_test, y_test (pd.DataFrame): The data frames to save.
    - config_path (str): Path to the configuration file.
    
    Returns:
    - None
    """
    
    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Create the base path using current datetime
    datetimerun = datetime.now().strftime("%Y-%m-%d-%H-%M")
    base_path = os.path.join(config['WORK_DIR'], config["DATA_ROOT"],'split', datetimerun)
    
    # Create the base and subdirectories if they do not exist
    subdirs = ['train', 'validation', 'test']
    for subdir in subdirs:
        path = os.path.join(base_path, subdir)
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Convert NumPy arrays to DataFrames
    x_train_df = pd.DataFrame(x_train)
    y_train_df = pd.DataFrame(y_train, columns=['target'])
    
    x_validation_df = pd.DataFrame(x_validation)
    y_validation_df = pd.DataFrame(y_validation, columns=['target'])
    
    x_test_df = pd.DataFrame(x_test)
    y_test_df = pd.DataFrame(y_test, columns=['target'])
    
    # Save the DataFrames
    x_train_df.to_csv(os.path.join(base_path, 'train', 'x_train.csv'), index=False)
    y_train_df.to_csv(os.path.join(base_path, 'train', 'y_train.csv'), index=False)
    
    x_validation_df.to_csv(os.path.join(base_path, 'validation', 'x_validation.csv'), index=False)
    y_validation_df.to_csv(os.path.join(base_path, 'validation', 'y_validation.csv'), index=False)
    
    x_test_df.to_csv(os.path.join(base_path, 'test', 'x_test.csv'), index=False)
    y_test_df.to_csv(os.path.join(base_path, 'test', 'y_test.csv'), index=False)
    
if __name__ == '__main__':
    args = parser.parse_args()
    df = read_csv(args.config)
    transform_df, sub_categories = transform(df)
    feature_engineering_df = feature_engineering(transform_df, sub_categories)
    x_train, y_train, x_validation, y_validation, x_test, y_test, vec = split_dataframe(feature_engineering_df, target_column='Price')
    save_data(x_train, y_train, x_validation, y_validation, x_test, y_test, args.config)

    estim = HyperoptEstimator(regressor=any_regressor('reg'),
                                algo=tpe.suggest, 
                                trial_timeout=300,
                          )

    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:
        estim.fit(x_train, y_train)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(vec, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    mlflow.sklearn.log_model(estim, artifact_path="models_mlflow")


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