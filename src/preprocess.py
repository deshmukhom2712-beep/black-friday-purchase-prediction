import pandas as pd, numpy as np

AGE_MAP = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}

def load_and_clean(path):
    df = pd.read_csv(path)
    # frequency encodings (no target leakage)
    df["User_Freq"] = df.groupby("User_ID")["User_ID"].transform("count")
    df["Product_Freq"] = df.groupby("Product_ID")["Product_ID"].transform("count")
    # missing product categories -> 0 means "no category"
    for c in ["Product_Category_2", "Product_Category_3"]:
        df[c] = df[c].fillna(0).astype(int)
    df["Gender"] = (df["Gender"] == "M").astype(int)
    df["Age"] = df["Age"].map(AGE_MAP)
    df["Stay_In_Current_City_Years"] = df["Stay_In_Current_City_Years"].astype(str).str.replace("+", "", regex=False).astype(int)
    df = pd.get_dummies(df, columns=["City_Category"], dtype=int)
    y = df.pop("Purchase")
    X = df.drop(columns=["User_ID", "Product_ID"])
    return X, y
