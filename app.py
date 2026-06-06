import os
# Force CPU execution before any TensorFlow/Keras imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Configure Streamlit page
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

@st.cache_resource
def load_artifacts():
    # Load the scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # Load the model
    model = tf.keras.models.load_model("models/trained_diabetes_model.keras")
    return scaler, model

scaler, model = load_artifacts()

# UI Layout Architecture - Sidebar
st.sidebar.header("Patient Medical Data")
st.sidebar.markdown("Please input the patient's data below:")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
glucose = st.sidebar.slider("Glucose (mg/dL)", min_value=40.0, max_value=200.0, value=100.0, step=1.0)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg)", min_value=20.0, max_value=140.0, value=70.0, step=1.0)
skin_thickness = st.sidebar.slider("Skin Thickness (mm)", min_value=5.0, max_value=100.0, value=20.0, step=1.0)
insulin = st.sidebar.slider("Insulin (IU/mL)", min_value=10.0, max_value=900.0, value=79.0, step=1.0)
bmi = st.sidebar.slider("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.50, value=0.50, step=0.01)
age = st.sidebar.number_input("Age (years)", min_value=21, max_value=100, value=30, step=1)

# UI Layout Architecture - Main Area
st.title("🩺 Diabetes Risk Predictor")
st.markdown("""
This application utilizes a trained Artificial Neural Network (ANN) to predict the likelihood of a patient having diabetes based on the Pima Indians Diabetes Database metrics.

Adjust the patient medical data from the sidebar, and click **Predict** below.
""")

if st.button("Predict Risk", type="primary"):
    # Convert input to 2D numpy array
    input_features = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
    ]])
    
    # Scale the features
    scaled_features = scaler.transform(input_features)
    
    # Predict the probability
    prediction_prob = model.predict(scaled_features, verbose=0)[0][0]
    
    # Display result
    st.subheader("Prediction Result")
    
    confidence_percentage = prediction_prob * 100
    
    if prediction_prob >= 0.5:
        st.error("⚠️ **High Risk of Diabetes**")
        st.write(f"The model predicts a high likelihood of diabetes with a risk probability of **{confidence_percentage:.2f}%**.")
    else:
        st.success("✅ **Low Risk of Diabetes**")
        st.write(f"The model predicts a low likelihood of diabetes with a risk probability of **{confidence_percentage:.2f}%**.")
