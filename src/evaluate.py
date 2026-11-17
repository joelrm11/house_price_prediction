
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_processing import load_data, split_features_target
import joblib


def evaluate_model():
    # Load dataset and trained model
    df = load_data("data/train.csv")
    model = joblib.load("models/house_price_model.joblib")

    # Separate features and target
    X, y = split_features_target(df)

    # Recreate the same test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Evaluate on the test set
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Test MAE: ${mae:,.2f}")
    print(f"Test R²: {r2:.3f}")

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    cv_mae = -cv_scores

    print("\nCross-validation MAE scores:")
    for i, score in enumerate(cv_mae, 1):
        print(f"Fold {i}: ${score:,.2f}")

    print(f"Average CV MAE: ${cv_mae.mean():,.2f}")


if __name__ == "__main__":
    evaluate_model()
