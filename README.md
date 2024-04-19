# Wind Speed Prediction

This project aims to predict wind speed using various machine learning models. The project is implemented in Python and uses several libraries such as pandas, matplotlib, seaborn, and scikit-learn.

## Project Structure

The project consists of two main Python scripts:

1. `EDA.py`: This script is used for exploratory data analysis. It includes functions for reading the dataset, plotting feature distributions, correlation heatmaps, and scatter matrices.

2. `ml.py`: This script is used for training and evaluating machine learning models. It includes functions for splitting the dataset, preprocessing the data, defining the machine learning pipelines, and evaluating the models.

## Setup

To set up the project, follow these steps:

1. Clone the repository.
2. Install the required Python packages using pip:

    ```
    pip install -r requirements.txt
    ```

3. Run the `EDA.py` script to perform exploratory data analysis:

    ```
    python EDA.py
    ```

4. Run the `ml.py` script to train and evaluate the machine learning models:

    ```
    python ml.py
    ```

## Data

The dataset used in this project is stored in the `data` folder. The dataset is a CSV file named `wind_dataset.csv`.

## Models

The project uses several machine learning models including Ridge Regression, Gradient Boosting, Lasso, ElasticNet, and MLPRegressor. The models are evaluated using R2 score, Mean Relative Percentage Error, and Root Mean Squared Error.

## License

This project is open source and available under the [MIT License](LICENSE).