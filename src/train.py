
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_processing import load_data, split_features_target


def train_model():
    # Load dataset
    df = load_data("data/train.csv")

    # Separate features and target
    X, y = split_features_target(df)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify numerical and categorical features
    numeric_features = X_train.select_dtypes(include="number").columns
    categorical_features = X_train.select_dtypes(exclude="number").columns

    # Prepare numerical data
    numeric_transformer = SimpleImputer(strategy="median")

    # Prepare categorical data
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Create the complete model pipeline
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Absolute Error: ${mae:,.2f}")
    print(f"R² Score: {r2:.3f}")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/house_price_model.joblib")

    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()
