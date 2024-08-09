import streamlit as st
import pickle
import joblib  # assuming joblib was used for saving the model
import numpy as np
import pandas as pd

# Title of the app
st.title("Model Prediction App")

# File uploader for the model file
model_file = st.file_uploader("Upload the trained model file", type=["pkl"])
data_file = st.file_uploader("Upload the CSV file for prediction", type=["csv"])

if model_file and data_file:
    # Load the model
    loaded_model = joblib.load(model_file)
    
    # Load the new data
    new_data = pd.read_csv(data_file)
    original = new_data['Buy Box Seller']

    # Preprocessing
    new_data.drop(columns=['ASIN', 'open_date', 'item_is_marketplace', 'Buy Box Seller'], axis=1, inplace=True)
    new_data['Last Price Change'] = pd.to_datetime(new_data['Last Price Change'])
    new_data['Price Change Month'] = new_data['Last Price Change'].dt.month
    new_data['Price Change Day'] = new_data['Last Price Change'].dt.day

    # Transform the data using the same transformations as the training data
    new_data_transformed = loaded_model.named_steps['column_trans'].transform(new_data)

    # Make predictions
    predictions = loaded_model.named_steps['grad_model'].predict(new_data_transformed)

    # Create a results dataframe
    result = pd.DataFrame(predictions, columns=['predictions'])
    result['original'] = original

    # Compare the predictions with the original values to see which ones are correct
    result['Correct'] = result['predictions'] == result['original']

    # Count how many predictions were correct
    correct_count = result['Correct'].sum()

    # Calculate the accuracy
    accuracy = correct_count / len(result) * 100

    # Display the results
    st.write("Number of correct predictions out of {}: {}".format(len(result), correct_count))
    st.write("Accuracy: {:.2f}%".format(accuracy))

    # Show the detailed results in a table
    st.write("Detailed Results")
    st.dataframe(result)

else:
    st.write("Please upload both the model file and the CSV data file.")
