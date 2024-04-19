import matplotlib.pyplot as plt
import pandas as pd
from EDA import read_csv_from_data_folder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import PredictionErrorDisplay
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor

import numpy as np


def mean_relative_percentage_error(y_true, y_pred):
    return np.mean(np.where(y_true != 0, np.abs((y_true - y_pred) / y_true), 0)) * 100


def train_test_val_split(df: pd.DataFrame, target_column: str, test_frac: float = 0.2) -> tuple:
    """
    Split the DataFrame into training, testing, and validation sets.

    Parameters:
    df (pd.DataFrame): The DataFrame to split.
    target_column (str): The name of the target column.

    test_frac (float): The fraction of the data to be used for testing.

    Returns:
    tuple: The training, testing, and validation sets as DataFrames.
    """

    x = df.drop(columns=[target_column])
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac, random_state=42)

    return x_train, y_train, x_test, y_test


class DateSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='DATE'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X = X.copy()  # to avoid changes to the original dataframe
        X['Month'] = X[self.date_column].str.split('-', expand=True)[1]
        X = X.drop(columns=[self.date_column])
        return X


def main() -> None:
    df = read_csv_from_data_folder("wind_dataset.csv")

    x_train, y_train, x_test, y_test = train_test_val_split(df, 'WIND')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    numerical_features = df.columns.drop('DATE').drop('WIND')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, ['Month'])
        ])

    pipe1 = Pipeline([
        ('date_splitter', DateSplitter()),
        ('preprocessor', preprocessor),
        ('regressor', Ridge())  # Use Ridge regressor here
    ])

    pipe2 = Pipeline([
        ('date_splitter', DateSplitter()),
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])

    pipe3 = Pipeline([
        ('date_splitter', DateSplitter()),
        ('preprocessor', preprocessor),
        ('regressor', Lasso())
    ])

    pipe4 = Pipeline([
        ('date_splitter', DateSplitter()),
        ('preprocessor', preprocessor),
        ('regressor', ElasticNet())  # Use ElasticNet regressor here
    ])
    pipe5 = Pipeline([
        ('date_splitter', DateSplitter()),
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(random_state=42, max_iter=5000))  # Use MLPRegressor here
    ])

    voting_regressor = VotingRegressor([
        # ('ridge', pipe1),
        # ('gradient_boosting', pipe2),
        # ('lasso', pipe3),
        # ('elastic_net', pipe4),
        ('mlp', pipe5)  #
    ])

    param_dist = {
        # 'ridge__regressor__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],  # Define the hyperparameters for Ridge
        # 'gradient_boosting__regressor__n_estimators': range(50, 300),
        # 'gradient_boosting__regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        # 'gradient_boosting__regressor__max_depth': range(3, 10),
        # 'gradient_boosting__regressor__min_samples_split': range(2, 10),
        # 'gradient_boosting__regressor__min_samples_leaf': range(1, 10),
        # 'lasso__regressor__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],  # Define the hyperparameters for Lasso
        # 'elastic_net__regressor__alpha': [0.1, 0.5, 1.0, 2.0, 5.0],  # Define the hyperparameters for ElasticNet
        # 'elastic_net__regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # Define the hyperparameters for ElasticNet
        'mlp__regressor__hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Define the hyperparameters for MLPRegressor
        'mlp__regressor__activation': ['relu', 'tanh'],  # Define the hyperparameters for MLPRegressor
        'mlp__regressor__solver': ['adam', 'sgd'],  # Define the hyperparameters for MLPRegressor
        'mlp__regressor__alpha': [0.0001, 0.001, 0.01],  # Define the hyperparameters for MLPRegressor
        'mlp__regressor__learning_rate': ['constant', 'invscaling', 'adaptive']

    }

    random_search = RandomizedSearchCV(voting_regressor, param_distributions=param_dist, n_iter=50, cv=5, n_jobs=-1,
                                       random_state=42)
    random_search.fit(x_train, y_train)

    print("Best parameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)
    display = PredictionErrorDisplay.from_estimator(random_search, x_test, y_test, kind='actual_vs_predicted')
    display.plot()
    plt.show()

    y_pred_test = random_search.predict(x_test)
    print("R2 score test data: ", r2_score(y_test, y_pred_test))
    print("Mean Relative Percentage Error test data: ", mean_relative_percentage_error(y_test, y_pred_test))
    print("Root Mean Squared Error test data: ", np.sqrt(np.mean((y_test - y_pred_test) ** 2)))

    y_pred_train = random_search.predict(x_train)
    print("R2 score train data: ", r2_score(y_train, y_pred_train))
    print("Mean Relative Percentage Error train data: ", mean_relative_percentage_error(y_train, y_pred_train))
    print("Root Mean Squared Error train data: ", np.sqrt(np.mean((y_train - y_pred_train) ** 2)))


if __name__ == "__main__":
    main()
