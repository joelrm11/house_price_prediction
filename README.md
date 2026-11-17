# House Price Prediction Using Machine Learning

## 1. Project Overview

This project uses Machine Learning to predict house sale prices based on different housing characteristics.

The project uses the Ames Housing dataset from the Kaggle competition:

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

The main algorithm used is the Random Forest Regressor.

## 2. Project Objectives

- Explore and understand housing data.
- Perform data cleaning and preprocessing.
- Train a Machine Learning model to predict house prices.
- Evaluate model performance.
- Identify the most important features affecting predictions.
- Build an interactive house price prediction dashboard.
- Generate predictions for the Kaggle test dataset.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Gradio
- Google Colab

## 4. Dataset

The project uses the Ames Housing dataset from Kaggle.

The dataset contains information about houses, including:

- Overall house quality
- Living area
- Basement area
- Garage capacity
- Year built
- Other property characteristics

The training dataset contains 1,460 records and the Kaggle test dataset contains 1,459 records.

Download the datasets from Kaggle and place `train.csv` and `test.csv` inside the `data` folder.

## 5. Project Structure

```text
house_price_prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── house_price_model.joblib
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── House_Price_Prediction_Final_Project.ipynb
├── demo.py
├── requirements.txt
├── submission.csv
├── .gitignore
└── README.md
```

Note: The dataset CSV files are excluded from GitHub using `.gitignore`. Download them from Kaggle before running the Python scripts.

## 6. Data Preprocessing

The following preprocessing techniques were applied:

- Numerical missing values were filled using median imputation.
- Categorical missing values were filled with the value `Missing`.
- Categorical variables were converted into numerical values using One-Hot Encoding.
- A Scikit-learn preprocessing pipeline was used to combine these steps.

## 7. Model Training

The project uses the Random Forest Regressor algorithm.

The model was configured with:

- 100 decision trees (`n_estimators=100`)
- Random state of 42
- Parallel processing using all available CPU cores

The dataset was divided into:

- 80% training data
- 20% testing data

The trained model was saved as:

`models/house_price_model.joblib`

## 8. Model Evaluation

The model was evaluated using Mean Absolute Error (MAE), R-squared (R²), and 5-fold cross-validation.

### Model Results

| Metric | Result |
|---|---:|
| Test MAE | $17,818.64 |
| Test R² | 0.887 |
| Average Cross-Validation MAE | $18,298.65 |

The results indicate that the Random Forest model performs well in predicting house prices on the test split.

## 9. Feature Importance

Feature importance analysis identified the most influential features used by the model.

The top features include:

1. OverallQual
2. GrLivArea
3. TotalBsmtSF
4. 2ndFlrSF
5. BsmtFinSF1

OverallQual (overall house quality) was the most important feature in the model.

Feature importance represents predictive contribution and does not establish causation.

## 10. Interactive Prediction Dashboard

An interactive dashboard was developed using Gradio.

Users can enter five house characteristics:

- Overall Quality
- Above-ground Living Area
- Total Basement Area
- Garage Capacity
- Year Built

The dashboard uses the trained model to estimate the house selling price. Other characteristics are set to typical values from the training dataset.

To launch the dashboard, run:

```bash
python demo.py
```

## 11. Installation and Usage

### Step 1: Clone the repository

```bash
git clone https://github.com/joelrm11/house_price_prediction.git
cd house_price_prediction
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the dataset

Download `train.csv` and `test.csv` from Kaggle and place them inside the `data` folder.

### Step 4: Train the model

```bash
python -m src.train
```

### Step 5: Evaluate the model

```bash
python -m src.evaluate
```

### Step 6: Launch the dashboard

```bash
python demo.py
```

## 12. Generating Kaggle Test Predictions

The project can also predict prices for the separate Kaggle test dataset.

Run:

```bash
python -m src.predict
```

The predictions are saved in:

`submission.csv`

The generated file contains two columns:

- `Id`
- `SalePrice`

It contains 1,459 predicted house prices and follows the required Kaggle submission format.

## 13. Conclusion

This project demonstrates how Machine Learning can be used to predict house selling prices using historical housing data.

The Random Forest Regressor achieved a test R² score of 0.887 and a Mean Absolute Error of approximately $17,819.

The project also includes feature importance analysis, an interactive Gradio dashboard, and predictions for the Kaggle test dataset.
