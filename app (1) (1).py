import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('iris_model.pkl')

st.title("Machine Learning on Iris Data")

st.write("Enter the sepal and petal measurements to predict the Iris species.")

# Input fields for sepal and petal measurements
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.5)

# Prediction button
if st.button("Predict Species"):
    input_data = pd.DataFrame([{
        'SepalLengthCm': sepal_length,
        'SepalWidthCm': sepal_width,
        'PetalLengthCm': petal_length,
        'PetalWidthCm': petal_width
    }])

    prediction = model.predict(input_data)
    st.success(f"The predicted Iris species is: {prediction[0]}")
