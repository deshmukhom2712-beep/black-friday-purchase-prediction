# Comparative Analysis of Predictive Learning Algorithms for Consumer Purchase Amount Prediction

Compares Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost
(+ tuned) and LightGBM on a large e-commerce dataset (target: `Purchase`).

## 1. Download the dataset (needs a free Kaggle login)
Kaggle: "Black Friday" by Sudhir Dolezel (search "Black Friday sales dataset", file `train.csv`, ~550,000 rows).
Alternative: "Black Friday Sales Prediction" on Analytics Vidhya (same data).
Put the file at:  `data/train.csv`

Expected columns: User_ID, Product_ID, Gender, Age, Occupation, City_Category,
Stay_In_Current_City_Years, Marital_Status, Product_Category_1/2/3, Purchase

## 2. Setup
    python -m venv venv
    venv\Scripts\activate          (Windows)   |   source venv/bin/activate  (Mac/Linux)
    pip install -r requirements.txt

## 3. Run
    python run.py --demo     # test the setup with synthetic data (2-3 min)
    python run.py --fast     # real data, no tuning, quick
    python run.py            # real data, full run (CV + XGBoost tuning, ~15-30 min)

## 4. Outputs (in `outputs/`)
- model_comparison.csv  -> results table for the report/PPT
- 00_purchase_distribution.png, 01_rmse_comparison.png, 02_r2_comparison.png,
  03_actual_vs_predicted.png, 04_feature_importance.png -> screenshots/charts
- best_xgb_params.json  -> tuned hyperparameters

Note: `--demo` data is synthetic and only checks that the code works. Report only results from the real dataset.
