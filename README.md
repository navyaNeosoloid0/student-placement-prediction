# 🎓 Student Placement Prediction System

An end-to-end machine learning project that predicts whether a student is likely to be placed,
from academic and skill data, and serves predictions through an interactive Streamlit app.

## Features
- Data cleaning (median imputation for missing values, one-hot encoding, scaling)
- EDA charts and a correlation heatmap
- Compares 3 models: Logistic Regression, Random Forest, XGBoost (Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold CV)
- Automatically saves the best model by F1 score
- Feature importance and confusion matrix
- Streamlit app with placement probability and personalised improvement tips

## Tech stack
Python, Pandas, NumPy, scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit, Joblib

## Project structure
```
placement-prediction/
├── data/placement_data.csv     # dataset
├── generate_data.py            # creates the synthetic dataset
├── train.py                    # cleaning + EDA + training + evaluation
├── app.py                      # Streamlit web app
├── models/                     # saved model + metadata
├── reports/                    # charts and model comparison table
└── requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
python generate_data.py     # skip if you use your own dataset
python train.py
streamlit run app.py
```

## Using a real Kaggle dataset
Download a campus placement dataset, and save it as `data/placement_data.csv` with these columns:
`branch, cgpa, internships, projects, certifications, aptitude_score, coding_score,
communication_score, backlogs, placed` (placed = 1/0). Rename or edit the `NUM`/`CAT` lists
in `train.py` if your columns differ, then run `python train.py`.

## Deploy free
1. Push this folder to GitHub (include `models/` and `reports/`).
2. Go to share.streamlit.io, connect the repo, set main file to `app.py`.

## Results (synthetic data)
See `reports/model_comparison.csv`. CGPA, aptitude, coding score and backlogs are the most important features.
> Note: the dataset is synthetic, so the numbers demonstrate the pipeline and are not real-world placement statistics.
