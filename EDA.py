import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FOLDER = "data"
FILE_NAME = "wind_dataset.csv"


def read_csv_from_data_folder(filename):
    """
    Read a CSV file from the data folder
    :param filename:
    :return:
    """

    file_path = f"{DATA_FOLDER}/{filename}"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

    return df


def plot_feature_distribution(df: pd.DataFrame):
    """
    Plot the distribution of each feature in the DataFrame
    :param df:
    :return:
    """
    df.hist(figsize=(10, 10))
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
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


def main():
    """
    Main function
    :return:
    """

    df = read_csv_from_data_folder(FILE_NAME)
    if df is not None:
        print(df.head())
        print(df.info())
        print(df.describe())
        plot_feature_distribution(df)
        plot_correlation_heatmap(df)


# This conditional is used to check whether this script is being run directly
if __name__ == "__main__":
    main()
