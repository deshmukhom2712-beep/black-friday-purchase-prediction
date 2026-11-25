import os, joblib
import pandas as pd, streamlit as st

st.set_page_config(page_title="Purchase Amount Predictor", page_icon="🛍️", layout="wide")
AGE_MAP = {"0-17": 0, "18-25": 1, "26-35": 2, "36-45": 3, "46-50": 4, "51-55": 5, "55+": 6}

@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib") if os.path.exists("models/model.joblib") else None

bundle = load_model()
st.title("🛍️ Consumer Purchase Amount Predictor")
st.caption("Predicts how much a customer will spend, using the best model from the comparative analysis (XGBoost).")

if bundle is None:
    st.error("Model not found. Run:  python src/train_model.py   (needs data/train.csv), then restart the app.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model comparison", "ℹ️ About"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Customer")
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.selectbox("Age group", list(AGE_MAP), index=2)
        occ = st.number_input("Occupation code (0-20)", 0, 20, 4)
        marital = st.selectbox("Marital status", [0, 1], format_func=lambda v: "Married" if v else "Unmarried")
    with c2:
        st.subheader("Location")
        city = st.selectbox("City category", ["A", "B", "C"], index=1)
        stay = st.selectbox("Years in current city", ["0", "1", "2", "3", "4+"], index=1)
    with c3:
        st.subheader("Product")
        cat1 = st.number_input("Product category 1 (1-18)", 1, 18, 5)
        cat2 = st.number_input("Product category 2 (0 = none)", 0, 18, 0)
        cat3 = st.number_input("Product category 3 (0 = none)", 0, 18, 0)

    with st.expander("Advanced (optional)"):
        st.write("How often this user / product appears in the sales data. Defaults are typical values.")
        uf = st.slider("User purchase count", 1, 1100, int(bundle["median_user_freq"]))
        pf = st.slider("Product purchase count", 1, 1900, int(bundle["median_product_freq"]))

    if st.button("Predict purchase amount", type="primary"):
        row = {"Gender": int(gender == "M"), "Age": AGE_MAP[age], "Occupation": occ,
               "Stay_In_Current_City_Years": int(stay.replace("+", "")), "Marital_Status": marital,
               "Product_Category_1": cat1, "Product_Category_2": cat2, "Product_Category_3": cat3,
               "User_Freq": uf, "Product_Freq": pf,
               "City_Category_A": int(city == "A"), "City_Category_B": int(city == "B"),
               "City_Category_C": int(city == "C")}
        df = pd.DataFrame([row])[bundle["columns"]]
        pred = float(bundle["model"].predict(df)[0])
        mae = bundle["metrics"]["mae"]
        st.metric("Predicted purchase amount", f"₹ {pred:,.0f}")
        st.info(f"Typical error of this model is about ±₹{mae:,.0f}, so expect roughly "
                f"₹{max(pred-mae,0):,.0f} to ₹{pred+mae:,.0f}.")

with tab2:
    if os.path.exists("outputs/model_comparison.csv"):
        res = pd.read_csv("outputs/model_comparison.csv")
        st.dataframe(res, width="stretch", hide_index=True)
        st.bar_chart(res.set_index("Model")["Test_R2"])
        cols = st.columns(2)
        for i, f in enumerate(["01_rmse_comparison.png", "03_actual_vs_predicted.png",
                               "04_feature_importance.png", "00_purchase_distribution.png"]):
            if os.path.exists(f"outputs/{f}"):
                cols[i % 2].image(f"outputs/{f}")
    else:
        st.warning("Run python run.py first to generate the comparison results.")

with tab3:
    m = bundle["metrics"]
    st.write(f"**Dataset:** Black Friday sales, {m['rows']:,} purchases.  \n"
             "**Task:** regression, predict the `Purchase` amount.  \n"
             "**Models compared:** Linear, Ridge, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM.")
    a, b, c = st.columns(3)
    a.metric("Hold-out R²", f"{m['r2']:.3f}"); b.metric("RMSE", f"{m['rmse']:,.0f}"); c.metric("MAE", f"{m['mae']:,.0f}")
