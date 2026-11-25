@echo off
python -m pip install -r requirements.txt
if not exist models\model.joblib python src\train_model.py
python webapp\app.py
pause
