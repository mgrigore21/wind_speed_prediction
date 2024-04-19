import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pandas.plotting import scatter_matrix

DATA_FOLDER = "data"
FILE_NAME = "wind_dataset.csv"


def read_csv_from_data_folder(filename: str) -> pd.DataFrame:
    """
    Read a CSV file from the data folder
    :param filename:
    :return:
    """

    file_path = f"{DATA_FOLDER}/{filename}"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return None

    return df


def plot_feature_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of each feature in the DataFrame
    :param df:
    :return:
    """
    df.hist(figsize=(10, 10))
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Plot the correlation heatmap of the DataFrame
    :param df:
    :return:
    """
    df_without_date = df.drop('DATE', axis=1)
    corr = df_without_date.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_scatter_matrix(df: pd.DataFrame) -> None:
    df_without_date = df.drop('DATE', axis=1)
    scatter_matrix(df_without_date, alpha=0.2, figsize=(10, 10), diagonal='kde')
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main function
    :return:
    """

    df = read_csv_from_data_folder(FILE_NAME)
    if df is not None:
        logging.info(df.head())
        logging.info(df.info())
        logging.info(df.describe())
        plot_feature_distribution(df)
        plot_correlation_heatmap(df)
        plot_scatter_matrix(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
