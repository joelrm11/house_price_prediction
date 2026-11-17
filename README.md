
# House Price Prediction

## 1. Project Overview

This project uses machine learning to predict house sale prices based on housing characteristics.

The project uses the Ames Housing dataset from the Kaggle competition:

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

The main machine learning algorithm is Random Forest Regression.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Gradio

## 3. Project Structure

```text
house_price_prediction/
│
├── data/
│   └── train.csv
│
├── models/
│   └── house_price_model.joblib
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   └── evaluate.py
│
├── demo.py
├── requirements.txt
└── README.md

## 12. Generating Kaggle Test Predictions

The project can also predict prices for the separate Kaggle test dataset.

1. Download `test.csv` from Kaggle.
2. Place it inside the `data` folder.
3. Run:

```bash
python -m src.predict
