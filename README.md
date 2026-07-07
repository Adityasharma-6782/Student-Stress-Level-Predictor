# Student Stress Level Predictor

This project predicts a student's **Stress Level (High/Low)** using lifestyle data — sleep hours, study hours, social media usage, attendance, exam pressure, and family support — through a Machine Learning model, deployed as a Streamlit web app.

## Project Structure

```
├── student-lifestyle-and-stress-dataset.csv   # Raw dataset
├── notebook.ipynb                              # Data cleaning, EDA, model training & comparison
├── save_model.py                                # Code to save the trained model as .pkl files
├── app.py                                       # Streamlit app (prediction UI)
├── requirements.txt                             # Python dependencies
├── stress_model.pkl                             # Trained best model (generated after running save_model.py)
├── scaler.pkl                                   # Fitted StandardScaler (generated)
├── model_columns.pkl                            # Training feature column order (generated)
└── num_cols.pkl                                 # Numeric column names for scaling (generated)
```

## Dataset Features

| Column | Description |
|---|---|
| Student_Type | school or college |
| Sleep_Hours | Daily sleep (hours) |
| Study_Hours | Daily study (hours) |
| Social_Media_Hours | Daily social media usage (hours) |
| Attendance | Attendance percentage (0-100) |
| Exam_Pressure | Exam pressure rating (0-10) |
| Family_Support | Family support rating (0-10) |
| Month | Month number (1-12) |
| Stress_Level | Target — 0 (Low) / 1 (High) |

## Workflow

1. **Data Cleaning** — fixed invalid values (negative Study_Hours, out-of-range Attendance), imputed missing values using median/mode, removed duplicates
2. **EDA** — explored feature relationships using a correlation heatmap
3. **Preprocessing** — one-hot encoding for categorical features, stratified train/test split, feature scaling (StandardScaler)
4. **Modeling** — Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest — tuned with GridSearchCV, optimized for F1-score (to handle class imbalance)
5. **Evaluation** — Accuracy, F1-score, Confusion Matrix, Classification Report; compared against the baseline (majority-class) accuracy
6. **Deployment** — best model, scaler, and column info saved as `.pkl` files; served via a Streamlit app for live predictions

## How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Train & save the model (if .pkl files aren't already present)
Run the notebook from top to bottom, then run the `save_model.py` code (last cell). This generates 4 `.pkl` files.

### 3. Run the app
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

### 4. Use the app
Enter your details (sleep hours, study hours, attendance, etc.) via the sliders/dropdown and click **Predict Stress Level** — the model will predict High or Low stress along with a confidence percentage.

## Model Performance Notes

- The target is imbalanced (~70% Low, ~30% High), so F1-score and the confusion matrix were prioritized over plain accuracy during evaluation
- `class_weight='balanced'` was used in applicable models so the minority class (High Stress) isn't ignored
- The final model was selected from `results_df` (sorted by F1-score) as the best performer

## Tech Stack

- Python, Pandas, NumPy, Scikit-learn
- Seaborn/Matplotlib (EDA)
- Streamlit (deployment)
- Joblib (model serialization)

## Future Improvements

- Collect more data to reduce class imbalance
- Feature engineering (interaction terms, e.g. Study_Hours × Exam_Pressure)
- Add explainability using SHAP/feature importance
- Cloud deployment (Streamlit Community Cloud / Render / HuggingFace Spaces)
