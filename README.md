# 🎓 Student Placement Prediction System

An end-to-end machine learning application that predicts student placement outcomes using academic performance, practical experience, and skill-related features. The project compares multiple classification algorithms and provides predictions and personalized improvement suggestions through an interactive Streamlit web application.

## 🚀 Features

- Data inspection and preprocessing
- Exploratory Data Analysis (EDA) with visualizations and correlation analysis
- Comparison of three machine learning models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, and cross-validation
- Saved model and metadata for application inference
- Feature importance and confusion matrix visualizations
- Interactive Streamlit prediction interface
- Placement probability estimates and personalized improvement recommendations

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Model Persistence:** Joblib

## 📂 Project Structure

```text
placement-prediction/
├── data/
│   └── train.csv                 # Training dataset (not committed)
├── models/
│   ├── best_model.joblib         # Saved trained model
│   └── meta.json                 # Model metadata and metrics
├── reports/
│   ├── confusion_matrix.png
│   ├── correlation.png
│   ├── eda.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   └── model_comparison.csv
├── app.py                        # Streamlit application
├── generate_data.py              # Data generation utility
├── train.py                      # Training and evaluation pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset and Features

The training pipeline uses a campus recruitment dataset sourced from `Krooz/Campus_Recruitment_CSV`.

The model uses the following input features:

- CGPA
- Internships
- Projects
- Workshops/Certifications
- Aptitude Test Score
- Soft Skills Rating
- Extracurricular Activities
- Placement Training
- SSC Marks
- HSC Marks

The target variable is `PlacementStatus`, with the classes `Placed` and `NotPlaced`.

The dataset is excluded from this repository. Ensure you have the dataset and permission to use it before running the training pipeline.

## 🤖 Model Evaluation

The models were evaluated using a held-out test set and grouped cross-validation to reduce leakage from identical feature profiles.

| Metric | Selected XGBoost Model |
|---|---:|
| Accuracy | 81.16% |
| Precision | 77.18% |
| Recall | 77.81% |
| F1-score | 77.49% |
| ROC-AUC | 88.47% |
| Mean CV F1-score | 75.88% |

XGBoost was selected by its F1-score on the held-out test set in the current training run. These results describe performance on this dataset and should not be interpreted as proof of equivalent real-world placement prediction accuracy.

See `reports/model_comparison.csv` for the model comparison results.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/navyaNeosoloid0/student-placement-prediction.git
cd student-placement-prediction
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application uses the saved model and metadata in the `models/` directory. You do not need to retrain the model just to run the application.

To retrain, first place the required training dataset at the path expected by `train.py`, then run:

```bash
python train.py
```

## ☁️ Deployment

The application can be deployed using Streamlit Community Cloud:

1. Push the project to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select this repository.
4. Set `app.py` as the application entry point.
5. Deploy the application.

Review dependency versions and verify that the saved model and report files are available before deploying.

## ⚠️ Limitations

- Model performance depends on the dataset's quality and representativeness.
- Test-set metrics do not guarantee performance on students from other institutions or recruitment settings.
- Placement predictions are estimates, not guarantees of employment.
- The model should support career guidance rather than be used as the sole basis for high-stakes recruitment decisions.

## 👩‍💻 Author

GitHub: [navyaNeosoloid0](https://github.com/navyaNeosoloid0)
