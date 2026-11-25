# Web app (Streamlit)

Extract these files INTO your existing purchase_prediction folder (overwrite requirements.txt).

    pip install -r requirements.txt
    python src/train_model.py        # trains final model on data/train.csv, saves models/model.joblib (~1 min)
    streamlit run app.py             # opens http://localhost:8501

Tabs: Predict (enter customer + product details), Model comparison (needs outputs/ from run.py), About.

## Put it online for free (Streamlit Community Cloud)
1. Push the project to a GitHub repo (include app.py, requirements.txt, models/model.joblib, outputs/*.csv and *.png; NOT data/train.csv).
2. Go to share.streamlit.io, sign in with GitHub, choose the repo and set main file = app.py.
3. Deploy. You get a public link to put in the report and PPT.
