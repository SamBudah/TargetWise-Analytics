
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Set page configuration for a wider layout and custom theme
st.set_page_config(page_title="TargetWise Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling (added fixed title)
st.markdown('''
<style>
/* Fixed title at the top */
.fixed-title {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: #f0f2f6;  /* Match the main background */
    color: #000000;  /* Black text */
    padding: 10px 20px;
    z-index: 1000;  /* Ensure it stays above other elements */
    font-size: 24px;
    font-weight: bold;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);  /* Subtle shadow for depth */
}

/* Add padding to the main content to avoid overlap with the fixed title */
.stApp {
    padding-top: 50px;  /* Adjust based on the height of the fixed title */
}

/* Main content area */
.main {
    background-color: #f0f2f6;
    color: #000000;  /* Black text */
}

/* Ensure the main content area background is applied */
.stApp {
    background-color: #f0f2f6;
    color: #000000;  /* Black text */
}

/* Buttons */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
}
.stButton>button:hover {
    background-color: #45a049;
}

/* Sidebar */
.stSidebar {
    background-color: #d3d8e8;
    color: #000000;  /* Black text */
}

/* Ensure sidebar text is visible */
.stSidebar .stMarkdown, .stSidebar .stSelectbox {
    color: #000000;  /* Black text */
}

/* Input fields and labels */
label, .stNumberInput, .stSlider, .stSelectbox {
    color: #000000;  /* Black text */
}

/* Ensure text in expanders and other components is black */
.stExpander, .stMarkdown, .stSuccess, .stWarning, .stError {
    color: #000000;  /* Black text */
}
</style>
''', unsafe_allow_html=True)

# Add the fixed title at the top of the page
st.markdown('<div class="fixed-title">TargetWise Analytics</div>', unsafe_allow_html=True)

# Sidebar for description and instructions (updated description)
st.sidebar.title("TargetWise Analytics")
st.sidebar.markdown("Built with KMeans and Random Forest")
st.sidebar.markdown('''
This app allows Cooperatives to predict customer segments in real-time.
- Enter customer details below.
- Click 'Predict Segment' to classify the customer.
- Use the insights to tailor marketing strategies.
''')

# Main content
st.markdown("Enter customer details to predict their segment and get marketing recommendations.")

# Load the trained Random Forest model and StandardScaler
try:
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("Model or scaler file not found. Please train and save the model first.")
    st.stop()

# Define the features used for prediction
numerical_features = ['reports', 'age', 'income', 'share', 'expenditure', 'dependents', 'months', 'majorcards', 'active']
categorical_features = ['card', 'owner', 'selfemp']

# Create a form for user inputs
with st.form(key='customer_form'):
    st.header("Enter Customer Details")

    # Create input fields for numerical features
    numerical_inputs = {}
    for feature in numerical_features:
        if feature == 'age':
            numerical_inputs[feature] = st.slider(f"{feature.capitalize()} (years)", min_value=18, max_value=100, value=30)
        elif feature == 'income':
            numerical_inputs[feature] = st.number_input(f"{feature.capitalize()} (in Ksh)", min_value=0.0, value=30000.0, step=1000.0)
        elif feature == 'share':
            numerical_inputs[feature] = st.number_input(f"{feature.capitalize()} (ratio of expenditure to income)", min_value=0.0, value=0.05, step=0.01)
        elif feature == 'expenditure':
            numerical_inputs[feature] = st.number_input(f"{feature.capitalize()} (monthly, in Ksh)", min_value=0.0, value=500.0, step=10.0)
        else:
            numerical_inputs[feature] = st.number_input(f"{feature.capitalize()}", min_value=0, value=0, step=1)

    # Create input fields for categorical features
    categorical_inputs = {}
    for feature in categorical_features:
        categorical_inputs[feature] = st.selectbox(f"{feature.capitalize()}", options=['yes', 'no'], index=1)

    # Add submit and reset buttons
    col1, col2 = st.columns(2)
    with col1:
        submit_button = st.form_submit_button(label="Predict Segment")
    with col2:
        reset_button = st.form_submit_button(label="Reset")

# Display the input data and make prediction
if submit_button:
    # Combine numerical and categorical inputs into a dictionary
    input_dict = {**numerical_inputs, **categorical_inputs}
    input_df_display = pd.DataFrame([input_dict])
    st.subheader("Input Data")
    st.dataframe(input_df_display)

    # Prepare the input data for prediction
    # Numerical features
    input_data = [numerical_inputs[feature] for feature in numerical_features]
    input_df = pd.DataFrame([input_data], columns=numerical_features)

    # Standardize numerical features
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])

    # Categorical features (convert to dummy variables)
    for feature in categorical_features:
        input_df[f"{feature}_yes"] = 1 if categorical_inputs[feature] == 'yes' else 0

    # Ensure the input DataFrame matches the training data (add missing dummy columns if needed)
    expected_columns = [col for col in rf_model.feature_names_in_ if col not in numerical_features]
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[rf_model.feature_names_in_]

    # Make prediction
    prediction = rf_model.predict(input_df)[0]

    # Display the result
    st.success(f"The customer belongs to **Cluster {prediction}**")

    # Display insights in an expander
    with st.expander("Insights", expanded=True):
        if prediction == 0:
            st.write("**Cluster 0**: Moderate precision and recall, indicating potential overlap with other segments. Marketing strategies should focus on distinguishing this segment more clearly.")
        elif prediction == 1:
            st.write("**Cluster 1**: High recall and precision, making it the most consistently identified group. This segment is highly predictable, possibly indicating loyal or consistent purchasing behavior.")
        elif prediction == 2:
            st.write("**Cluster 2**: Although the smallest group, it maintained high precision and recall, suggesting a niche but well-defined segment.")
        elif prediction == 3:
            st.write("**Cluster 3**: High precision, making it a distinct segment from others, ideal for targeted campaigns.")

    # Display recommendations in an expander
    with st.expander("Recommendations"):
        st.write('''
        - **Marketing Strategy**: Tailor marketing campaigns to target this segment more effectively, maximizing customer engagement and conversion rates.
        - **Business Decision-Making**: Utilize this segment for product recommendations, personalized offers, and strategic inventory management.
        ''')
