import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "employee_attrition_model.pkl"
)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("Employee Attrition Prediction")

st.subheader("Employee Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=60,
    value=41
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

job_role = st.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources"
    ]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=20000,
    value=5993
)

over18 = st.selectbox(
    "Over18",
    ["Y"]
)

overtime = st.selectbox(
    "OverTime",
    ["Yes", "No"]
)

if st.button("Predict Attrition"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "JobRole": [job_role],
        "MonthlyIncome": [monthly_income],
        "Over18": [over18],
        "OverTime": [overtime]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Yes, Employee will leave")
    else:
        st.error("No, Employee will stay")

    st.write(f"Stay Probability: {probability[0]:.2%}")
    st.write(f"Leave Probability: {probability[1]:.2%}")