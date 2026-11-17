
import pandas as pd


def load_data(file_path):
    """
    Load the housing dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df


def split_features_target(df):
    """
    Separate input features (X) from the target variable (y).
    """
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    return X, y
