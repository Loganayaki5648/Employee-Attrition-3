import streamlit as st
import pandas as pd
import pickle


df = pd.read_csv("Employee-Attrition.csv")

st.title("Employee Attrition Prediction App")

st.subheader("Employee Dataset Preview")
st.dataframe(df)


model = pickle.load(open("lr_scaled.pkl", "rb"))
preprocessor = pickle.load(open("preprocessor.pkl", "rb"))

st.subheader("Predict for New Employee")


age = st.slider("Age", 18, 60, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_map = {"Male": 1, "Female": 0}
input_gender = gender_map[gender]

job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Others"])

monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)

over18_input = st.selectbox("Over18", ["Y", "N"])
over18 = 1 if over18_input == "Y" else 0

overtime_input = st.selectbox("OverTime", ["Yes", "No"])
overtime = 1 if overtime_input == "Yes" else 0

input_df = pd.DataFrame({
    'Age': [age],
    'DailyRate': [1200],
    'DistanceFromHome': [5],
    'Education': [3],
    'EmployeeCount': [1],
    'EmployeeNumber': [1001],
    'EnvironmentSatisfaction': [3],
    'Gender': [input_gender],
    'HourlyRate': [70],
    'JobInvolvement': [3],
    'JobLevel': [2],
    'JobSatisfaction': [4],
    'MonthlyIncome': [monthly_income],
    'MonthlyRate': [14000],
    'NumCompaniesWorked': [2],
    'Over18': [over18],
    'OverTime': [overtime],
    'PercentSalaryHike': [15],
    'PerformanceRating': [3],
    'RelationshipSatisfaction': [3],
    'StandardHours': [80],
    'StockOptionLevel': [1],
    'TotalWorkingYears': [10],
    'TrainingTimesLastYear': [3],
    'WorkLifeBalance': [3],
    'YearsAtCompany': [5],
    'YearsInCurrentRole': [3],
    'YearsSinceLastPromotion': [1],
    'YearsWithCurrManager': [2],
    'BusinessTravel_Travel_Frequently': [0],
    'BusinessTravel_Travel_Rarely': [1],
    'Department_Research & Development': [1],
    'Department_Sales': [0],
    'EducationField_Life Sciences': [1],
    'EducationField_Marketing': [0],
    'EducationField_Medical': [0],
    'EducationField_Other': [0],
 'EducationField_Technical Degree': [0],
    'JobRole_Human Resources': [0],
    'JobRole_Laboratory Technician': [0],
    'JobRole_Manager': [0],
    'JobRole_Manufacturing Director': [0],
    'JobRole_Research Director': [0],
    'JobRole_Research Scientist': [1],
    'JobRole_Sales Executive': [0],
    'JobRole_Sales Representative': [0],
    'MaritalStatus_Married': [0],
    'MaritalStatus_Single': [1]
})

input_processed = preprocessor.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_df)
    result = "Yes, Employee will leave" if prediction[0] == 1 else "No, Employee will stay"
    st.subheader(f"Attrition Prediction: {result}")
    
