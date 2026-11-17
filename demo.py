
import joblib
import pandas as pd
import gradio as gr
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Load the trained model and original training data
model = joblib.load(ROOT / "models" / "house_price_model.joblib")
df = pd.read_csv(ROOT / "data" / "train.csv")

X = df.drop("SalePrice", axis=1)

# Create a typical house using median numerical values
# and the most common categorical values
base_house = X.median(numeric_only=True).to_frame().T

categorical_features = X.select_dtypes(exclude="number").columns

for col in categorical_features:
    base_house[col] = X[col].mode()[0]


def predict_price(overall_qual, gr_liv_area, total_bsmt_sf,
                  garage_cars, year_built):

    house = base_house.copy()

    house["OverallQual"] = overall_qual
    house["GrLivArea"] = gr_liv_area
    house["TotalBsmtSF"] = total_bsmt_sf
    house["GarageCars"] = garage_cars
    house["YearBuilt"] = year_built

    prediction = model.predict(house)[0]

    return f"Estimated house price: ${prediction:,.2f}"


demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(1, 10, value=5, step=1, label="Overall Quality"),
        gr.Number(value=1500, label="Above-ground Living Area (sq ft)"),
        gr.Number(value=800, label="Total Basement Area (sq ft)"),
        gr.Slider(0, 4, value=2, step=1, label="Garage Capacity (cars)"),
        gr.Slider(1870, 2010, value=2000, step=1, label="Year Built")
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="House Price Prediction",
    description=(
        "Enter five house characteristics to estimate its sale price. "
        "Other characteristics are set to typical values from the dataset."
    )
)


if __name__ == "__main__":
    demo.launch(share=True)
