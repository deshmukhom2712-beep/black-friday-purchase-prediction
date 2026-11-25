import argparse, time, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from preprocess import load_and_clean

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/train.csv")
p.add_argument("--out", default="outputs")
p.add_argument("--cv_sample", type=int, default=100_000, help="rows used for 5-fold CV and tuning")
p.add_argument("--fast", action="store_true", help="skip tuning and slow models for a quick test")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
SEED = 42

X, y = load_and_clean(a.data)
print(f"Rows: {len(X):,}  Features: {X.shape[1]}")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=SEED),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, n_jobs=-1, random_state=SEED),
    "LightGBM": LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.1, n_jobs=-1, random_state=SEED, verbose=-1),
}
if a.fast:
    models.pop("Gradient Boosting"); models["Random Forest"].set_params(n_estimators=30)

# 5-fold CV on a subsample (keeps runtime sensible on large data)
cv_idx = Xtr.sample(min(a.cv_sample, len(Xtr)), random_state=SEED).index
Xcv, ycv = Xtr.loc[cv_idx], ytr.loc[cv_idx]
kf = KFold(5, shuffle=True, random_state=SEED)

rows, preds, fitted = [], {}, {}
for name, m in models.items():
    t = time.time()
    cv_rmse = -cross_val_score(m, Xcv, ycv, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=1).mean()
    m.fit(Xtr, ytr)
    pr = m.predict(Xte)
    preds[name], fitted[name] = pr, m
    rows.append({"Model": name,
                 "CV_RMSE(5-fold)": round(cv_rmse, 2),
                 "Test_RMSE": round(np.sqrt(mean_squared_error(yte, pr)), 2),
                 "Test_MAE": round(mean_absolute_error(yte, pr), 2),
                 "Test_R2": round(r2_score(yte, pr), 4),
                 "Time_s": round(time.time() - t, 1)})
    print(rows[-1])

# Hyperparameter tuning for XGBoost
if not a.fast:
    print("Tuning XGBoost...")
    rs = RandomizedSearchCV(
        XGBRegressor(n_jobs=-1, random_state=SEED),
        {"n_estimators": [200, 300, 500], "max_depth": [5, 7, 9], "learning_rate": [0.03, 0.05, 0.1],
         "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.7, 0.9, 1.0]},
        n_iter=8, cv=3, scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1)
    rs.fit(Xcv, ycv)
    best = XGBRegressor(**rs.best_params_, n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    pr = best.predict(Xte)
    preds["XGBoost (Tuned)"], fitted["XGBoost (Tuned)"] = pr, best
    rows.append({"Model": "XGBoost (Tuned)", "CV_RMSE(5-fold)": round(-rs.best_score_, 2),
                 "Test_RMSE": round(np.sqrt(mean_squared_error(yte, pr)), 2),
                 "Test_MAE": round(mean_absolute_error(yte, pr), 2),
                 "Test_R2": round(r2_score(yte, pr), 4), "Time_s": np.nan})
    json.dump(rs.best_params_, open(f"{a.out}/best_xgb_params.json", "w"), indent=2, default=str)

res = pd.DataFrame(rows).sort_values("Test_RMSE").reset_index(drop=True)
res.to_csv(f"{a.out}/model_comparison.csv", index=False)
print("\n", res.to_string(index=False))
best_name = res.iloc[0]["Model"]; print("\nBest model:", best_name)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(9, 5)); sns.barplot(data=res, y="Model", x="Test_RMSE", color="#2a6fdb")
plt.title("Model comparison: Test RMSE (lower is better)"); plt.tight_layout()
plt.savefig(f"{a.out}/01_rmse_comparison.png", dpi=150); plt.close()
plt.figure(figsize=(9, 5)); sns.barplot(data=res, y="Model", x="Test_R2", color="#16a34a")
plt.title("Model comparison: Test R² (higher is better)"); plt.tight_layout()
plt.savefig(f"{a.out}/02_r2_comparison.png", dpi=150); plt.close()
idx = np.random.default_rng(SEED).choice(len(yte), min(3000, len(yte)), replace=False)
plt.figure(figsize=(6, 6)); plt.scatter(yte.values[idx], preds[best_name][idx], s=6, alpha=.4)
lim = [yte.min(), yte.max()]; plt.plot(lim, lim, "r--"); plt.xlabel("Actual Purchase"); plt.ylabel("Predicted Purchase")
plt.title(f"Actual vs Predicted ({best_name})"); plt.tight_layout()
plt.savefig(f"{a.out}/03_actual_vs_predicted.png", dpi=150); plt.close()
tree_res = res[res.Model.isin([m for m in fitted if hasattr(fitted[m], "feature_importances_")])]
fi_name = tree_res.iloc[0]["Model"]; bm = fitted[fi_name]
if True:
    fi = pd.Series(bm.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(8, 6)); fi.plot.barh(color="#f59e0b"); plt.title(f"Feature importance ({fi_name})")
    plt.tight_layout(); plt.savefig(f"{a.out}/04_feature_importance.png", dpi=150); plt.close()
plt.figure(figsize=(7, 4)); sns.histplot(y, bins=50, kde=True); plt.title("Distribution of Purchase amount")
plt.tight_layout(); plt.savefig(f"{a.out}/00_purchase_distribution.png", dpi=150); plt.close()
print("Charts and tables saved in", a.out)
