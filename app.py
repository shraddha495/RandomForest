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
    "Please fill in the customer details below to predict the outcome."
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
                "Student",
                "Self Employed",
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
    # --- MAPPING CATEGORICAL STRINGS TO NUMBERS ---
    # Update these numbers if your training script used a different mapping order!
    gender_map = {"Male": 1, "Female": 0, "Other": 2}
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
    occupation_map = {
        "Employee": 0,
        "Student": 1,
        "Self Employed": 2,
        "Housewife": 3,
        "Other": 4,
    }
    edu_map = {
        "School": 0,
        "Graduate": 1,
        "Post Graduate": 2,
        "Professional": 3,
        "Others": 4,
    }
    customer_type_map = {"New": 0, "Regular": 1, "VIP": 2}

    # Convert inputs to numeric form
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender_map.get(gender, 0)],
            "Marital Status": [marital_map.get(marital_status, 0)],
            "Occupation": [occupation_map.get(occupation, 0)],
            "Monthly Income": [monthly_income],
            "Educational Qualifications": [
                edu_map.get(educational_qualifications, 0)
            ],
            "Family size": [family_size],
            "Customer Type": [customer_type_map.get(customer_type, 0)],
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

        if result == "Yes" or result == 1:
            st.success(f"### Prediction: {result} 🎉", icon="✅")
        else:
            st.info(f"### Prediction: {result}", icon="ℹ️")

        if prediction_proba is not None:
            confidence = np.max(prediction_proba) * 100
            st.write(f"**Confidence Score:** {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
