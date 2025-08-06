import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import lime.lime_tabular

# --- Load model and encoders ---
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# --- Load and preprocess dataset ---
df = pd.read_csv("hospital_readmissions - Copy.csv")

for col in df.select_dtypes(include='object'):
    if col != 'readmitted':
        df[col] = label_encoders[col].transform(df[col])
df['readmitted'] = df['readmitted'].map({'no': 0, 'yes': 1})

X = df.drop("readmitted", axis=1)
y = df["readmitted"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Calculate metrics ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# --- Streamlit Sidebar ---
st.sidebar.title("Model Performance")
st.sidebar.metric("Accuracy", f"{acc * 100:.2f}%")
st.sidebar.write(f"Precision: {precision:.2f}")
st.sidebar.write(f"Recall: {recall:.2f}")

# --- LIME Explainer Setup ---
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X.head(500)),
    feature_names=X.columns.tolist(),
    class_names=["No Readmission", "Readmitted"],
    mode='classification'
)

# --- User Interface ---
st.title("Hospital Readmission Risk Predictor")
st.write("Enter patient data to assess readmission risk:")

user_input = {}
user_input['age'] = st.selectbox("Age", label_encoders['age'].classes_)
user_input['time_in_hospital'] = st.slider("Time in Hospital (days)", 1, 14, 5)
user_input['n_lab_procedures'] = st.slider("# Lab Procedures", 0, 100, 40)
user_input['n_procedures'] = st.slider("# Procedures", 0, 6, 1)
user_input['n_medications'] = st.slider("# Medications", 0, 50, 10)
user_input['n_outpatient'] = st.slider("# Outpatient Visits", 0, 20, 0)
user_input['n_inpatient'] = st.slider("# Inpatient Visits", 0, 20, 0)
user_input['n_emergency'] = st.slider("# Emergency Visits", 0, 20, 0)
user_input['medical_specialty'] = st.selectbox("Medical Specialty", label_encoders['medical_specialty'].classes_)
user_input['diag_1'] = st.selectbox("Primary Diagnosis", label_encoders['diag_1'].classes_)
user_input['diag_2'] = st.selectbox("Secondary Diagnosis", label_encoders['diag_2'].classes_)
user_input['diag_3'] = st.selectbox("Tertiary Diagnosis", label_encoders['diag_3'].classes_)
user_input['glucose_test'] = st.selectbox("Glucose Test", label_encoders['glucose_test'].classes_)
user_input['A1Ctest'] = st.selectbox("A1C Test", label_encoders['A1Ctest'].classes_)
user_input['change'] = st.selectbox("Medication Change", label_encoders['change'].classes_)
user_input['diabetes_med'] = st.selectbox("Diabetes Medication", label_encoders['diabetes_med'].classes_)

# --- Encode Input ---
encoded_input = []
encoded_input.append(label_encoders['age'].transform([user_input['age']])[0])
encoded_input.extend([
    user_input['time_in_hospital'],
    user_input['n_lab_procedures'],
    user_input['n_procedures'],
    user_input['n_medications'],
    user_input['n_outpatient'],
    user_input['n_inpatient'],
    user_input['n_emergency']
])

for col in ['medical_specialty', 'diag_1', 'diag_2', 'diag_3',
            'glucose_test', 'A1Ctest', 'change', 'diabetes_med']:
    encoded_val = label_encoders[col].transform([user_input[col]])[0]
    encoded_input.append(encoded_val)

# --- Predict & Explain ---
if st.button("Predict Readmission"):
    input_array = np.array(encoded_input).reshape(1, -1)
    proba = model.predict_proba(input_array)[0][1]
    st.subheader(f"Readmission Probability: {proba * 100:.2f}%")

    explanation = explainer.explain_instance(input_array[0], model.predict_proba, num_features=10)
    st.subheader("Top Contributing Factors")
    for feature, weight in explanation.as_list():
        st.write(f"{feature}: {weight:+.4f}")

