
import pandas as pd
import joblib


def predict_kaggle_test():
    # Load the trained model
    model = joblib.load("models/house_price_model.joblib")

    # Load Kaggle test dataset
    test_df = pd.read_csv("data/test.csv")

    # Keep Id as a feature because the model was trained with it
    predictions = model.predict(test_df)

    # Create submission file
    submission = pd.DataFrame({
        "Id": test_df["Id"],
        "SalePrice": predictions
    })

    submission.to_csv("submission.csv", index=False)

    print("Predictions completed!")
    print("Number of predictions:", len(submission))
    print("\nFirst five predictions:")
    print(submission.head())

    print("\nFile saved as submission.csv")


if __name__ == "__main__":
    predict_kaggle_test()
