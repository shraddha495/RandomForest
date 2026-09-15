import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Customer Prediction App", page_icon="📊", layout="centered"
)

# Custom CSS Style Code
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        margin-top: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the trained Random Forest model
@st.cache_resource
def load_model():
    with open("RandomForest.pkl", "rb") as file:
        model = pickle.load(file)
    return model


model = load_model()

# App Title and Description
st.title("Customer Behavior Prediction")
st.write(
    "Please fill in the customer details below to predict the outcome using your Random Forest model."
)

st.markdown("---")

# Input Form Layout
with st.form("prediction_form"):
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        marital_status = st.selectbox(
            "Marital Status", ["Single", "Married", "Divorced"]
        )
        occupation = st.selectbox(
            "Occupation",
            [
                "Employee",
                "Business",
                "Student",
                "Self-Employed",
                "Housewife",
                "Other",
            ],
        )

    with col2:
        monthly_income = st.number_input(
            "Monthly Income", min_value=0.0, value=50000.0, step=1000.0
        )
        educational_qualifications = st.selectbox(
            "Educational Qualifications",
            ["School", "Graduate", "Post Graduate", "Professional", "Others"],
        )
        family_size = st.number_input(
            "Family size", min_value=1, max_value=20, value=2
        )
        customer_type = st.selectbox("Customer Type", ["New", "Regular", "VIP"])

    submitted = st.form_submit_button("Predict Outcome")

# Prediction Logic
if submitted:
    # Create a DataFrame matching the model's expected features
    # Note: Ensure the order matches what your pipeline/model expects
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Marital Status": [marital_status],
            "Occupation": [occupation],
            "Monthly Income": [monthly_income],
            "Educational Qualifications": [educational_qualifications],
            "Family size": [family_size],
            "Customer Type": [customer_type],
        }
    )

    try:
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = (
            model.predict_proba(input_data)
            if hasattr(model, "predict_proba")
            else None
        )

        result = prediction[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        if result == "Yes":
            st.success(
                f"### Prediction: {result} 🎉", icon="✅"
            )  # Or customized output
        else:
            st.info(f"### Prediction: {result}", icon="ℹ️")

        if prediction_proba is not None:
            confidence = np.max(prediction_proba) * 100
            st.write(f"**Confidence Score:** {confidence:.2f}%")

    except Exception as e:
        st.error(
            f"Error during prediction: {e}. (Make sure categorical values match your training encoding scheme if encoders were part of a pipeline)."
        )
