"""Creates a synthetic dataset with the SAME columns as the Kaggle Black Friday data.
Use only to test the pipeline if you have not downloaded the real file."""
import numpy as np, pandas as pd, sys

def make(n=200_000, seed=42):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({
        "User_ID": r.integers(1000001, 1006041, n),
        "Product_ID": ["P00" + str(x) for x in r.integers(100000, 130000, n)],
        "Gender": r.choice(["M", "F"], n, p=[.75, .25]),
        "Age": r.choice(["0-17","18-25","26-35","36-45","46-50","51-55","55+"], n, p=[.03,.18,.40,.20,.08,.07,.04]),
        "Occupation": r.integers(0, 21, n),
        "City_Category": r.choice(["A","B","C"], n, p=[.27,.42,.31]),
        "Stay_In_Current_City_Years": r.choice(["0","1","2","3","4+"], n),
        "Marital_Status": r.integers(0, 2, n),
        "Product_Category_1": r.integers(1, 19, n),
    })
    df["Product_Category_2"] = np.where(r.random(n) < .31, np.nan, r.integers(2, 19, n))
    df["Product_Category_3"] = np.where(r.random(n) < .69, np.nan, r.integers(3, 19, n))
    base = 14000 - 650 * df.Product_Category_1 + (df.Gender == "M") * 400 \
           + df.City_Category.map({"A": 300, "B": 0, "C": 600})
    df["Purchase"] = np.clip(base + r.normal(0, 2500, n), 12, 23961).round().astype(int)
    return df

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/train.csv"
    make().to_csv(out, index=False); print("Saved", out)
