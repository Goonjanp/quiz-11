import streamlit as st
import pandas as pd
import pickle

# Load the trained Lasso model
filename = 'lasso_model.pkl'
lasso_model = pickle.load(open(filename, 'rb'))

# Create a title for your app
st.title("Monthly Revenue Prediction App")

# Add a brief description
st.write("This app predicts monthly revenue based on provided input features.")


# Create input fields for your features
# Example:
# st.number_input("Number of Products", min_value=0, max_value=1000, value=0)

# Get user input
# You need to adjust the input fields based on your features in X
# Here's an example of how to collect input for a few features:

new_customers = st.number_input("new_customers", value=0.0)
total_sku_count	 = st.number_input("total_sku_count	", value=0.0)
average_product_rating = st.number_input("average_product_rating", value=0.0)
# Add more input fields for your remaining features

# Prepare the input data as a DataFrame
input_data = pd.DataFrame({
    'new_customers': [new_customers],
    'total_sku_count': [total_sku_count],
    'average_product_rating': [average_product_rating],
    # Add more columns for your remaining features
})


# Make a prediction using the loaded Lasso model
if st.button("Predict"):
    prediction = lasso_model.predict(input_data)
    st.write(f"Predicted Monthly Revenue: {prediction[0]}")