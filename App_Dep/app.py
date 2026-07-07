import streamlit as st
import pandas as pd
import joblib

model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
num_cols = joblib.load("num_cols.pkl")

st.set_page_config(page_title="Student Stress Predictor", page_icon="🧠")
st.title("🧠 Student Stress Level Predictor")
st.write("Apni lifestyle details daalo aur dekho stress level predict hota hai ya nahi.")

# ---------- User inputs ----------
student_type = st.selectbox("Student Type", ["school", "college"])
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1)
study_hours = st.slider("Study Hours", 0.0, 15.0, 4.0, 0.1)
social_media_hours = st.slider("Social Media Hours", 0.0, 12.0, 3.0, 0.1)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 80.0, 0.1)
exam_pressure = st.slider("Exam Pressure (0-10)", 0.0, 10.0, 5.0, 0.5)
family_support = st.slider("Family Support (0-10)", 0.0, 10.0, 5.0, 0.5)
month = st.selectbox("Month", list(range(1, 13)))

# ---------- Predict button ----------
if st.button("Predict Stress Level"):

    input_df = pd.DataFrame([{
        "Student_Type": student_type,
        "Sleep_Hours": sleep_hours,
        "Study_Hours": study_hours,
        "Social_Media_Hours": social_media_hours,
        "Attendance": attendance,
        "Exam_Pressure": exam_pressure,
        "Family_Support": family_support,
        "Month": month
    }])

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    prediction = model.predict(input_encoded)[0]
    proba = model.predict_proba(input_encoded)[0] if hasattr(model, "predict_proba") else None

    st.subheader("Result")
    if prediction == 1:
        st.error("⚠️ High Stress Level predicted")
    else:
        st.success("✅ Low Stress Level predicted")

    if proba is not None:
        st.write(f"Confidence — Low Stress: {proba[0]:.2%}, High Stress: {proba[1]:.2%}")