import streamlit as st
import pandas as pd
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Set the header and sidebar title
st.header("Amazon Buy Box Prediction")
st.sidebar.title("Please select the features for prediction")

# Create input fields for each feature
price = st.sidebar.number_input("Price", min_value=0.0, value=19.99, step=0.01)
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
sales_rank_current = st.sidebar.slider("Sales Rank: Current", 0, 100000, step=5000)
reviews_rating = st.sidebar.slider("Reviews: Rating", 0.0, 5.0, value=4.5, step=0.1)
reviews_review_count = st.sidebar.number_input("Reviews: Review Count", min_value=0, value=100)
buy_box_stock = st.sidebar.selectbox("Buy Box 🚚: Stock", options=["In Stock", "Out of Stock"])
buy_box_is_fba = st.sidebar.selectbox("Buy Box: Is FBA", options=["Yes", "No"])

prime_eligible = st.sidebar.selectbox("Prime Eligible (Buy Box)", options=["Yes", "No"])
subscribe_save = st.sidebar.selectbox("Subscribe and Save", options=["Yes", "No"])
price_change_month = st.sidebar.slider("Price Change Month", 1, 12, value=6)
price_change_day = st.sidebar.slider("Price Change Day", 1, 31, value=15)

# Add a button to trigger the prediction
if st.sidebar.button("Predict"):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'price': [price],
        'quantity': [quantity],
        'Sales Rank: Current': [sales_rank_current],
        'Reviews: Rating': [reviews_rating],
        'Reviews: Review Count': [reviews_review_count],
        'Buy Box 🚚: Stock': [buy_box_stock],
        'Buy Box: Is FBA': [buy_box_is_fba],
        'Prime Eligible (Buy Box)': [prime_eligible],
        'Subscribe and Save': [subscribe_save],
        'Price Change Month': [price_change_month],
        'Price Change Day': [price_change_day]
    })

    # Transform the data
    column_trans = model.named_steps['column_trans']
    input_data_transformed = column_trans.transform(input_data)

    # Make predictions
    grad_model = model.named_steps['grad_model']
    prediction = grad_model.predict(input_data_transformed)

    # Display prediction result
    st.subheader("Prediction Results")
    st.write(f"The predicted Buy Box Seller is: **{prediction[0]}**")

