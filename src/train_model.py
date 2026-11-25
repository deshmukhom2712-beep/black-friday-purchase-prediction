"""Trains the final model (tuned XGBoost if available) and saves it for the web app."""
import argparse, os, json, joblib
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # always work from project root
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from preprocess import load_and_clean

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/train.csv")
a = p.parse_args()

X, y = load_and_clean(a.data)
params = dict(n_estimators=300, max_depth=7, learning_rate=0.1)
if os.path.exists("outputs/best_xgb_params.json"):
    params = json.load(open("outputs/best_xgb_params.json"))
    print("Using tuned parameters:", params)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
m = XGBRegressor(**params, n_jobs=-1, random_state=42).fit(Xtr, ytr)
pr = m.predict(Xte)
metrics = {"rmse": float(np.sqrt(mean_squared_error(yte, pr))),
           "mae": float(mean_absolute_error(yte, pr)), "r2": float(r2_score(yte, pr)),
           "rows": int(len(X))}
print("Hold-out metrics:", metrics)

final = XGBRegressor(**params, n_jobs=-1, random_state=42).fit(X, y)   # refit on all data
os.makedirs("models", exist_ok=True)
joblib.dump({"model": final, "columns": list(X.columns), "metrics": metrics,
             "median_user_freq": float(X.User_Freq.median()),
             "median_product_freq": float(X.Product_Freq.median())}, "models/model.joblib")
print("Saved models/model.joblib")
