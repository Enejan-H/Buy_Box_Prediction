import streamlit as st
import base64
import pickle
import pandas as pd
from PIL import Image
import os  # Import os module for handling file paths

# Function to set background image
def set_background(image_file):
    with open(image_file, "rb") as image:
        b64_encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{b64_encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Construct the path to the background image using os.path.join
image_path = os.path.join('1.png')

# Set the background image
set_background(image_path)

# Custom CSS for header background color only
st.markdown(
    """
    <style>
    .st-emotion-cache-12fmjuu {
        background-color: #ff9d3c;
    }
    
    .st-emotion-cache-1huvf7z,
    .st-emotion-cache-w3nhqi {
        background-color: transparent;
        color: inherit;
        border: none;
    }
    
    .st-emotion-cache-1wbqy5l span {
        color: inherit;
    }

    .st-emotion-cache-1pbsqtx {
        fill: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Adjust the heading with updated CSS
st.markdown("""
    <h1 style='text-align: center; color: #30373e; font-size: 55px; margin-top: -100px; margin-bottom: 0;'>Amazon Buy Box Prediction</h1>
    """, unsafe_allow_html=True)

# Load your images
img1 = os.path.join('emsbay.jpg')
img2 = os.path.join('whole-basket.png')
img3 = os.path.join('feature_importance.png')

# Display the first image at the top of the sidebar
st.sidebar.image(img1, use_column_width=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join('modell.pkl')  # Construct the path to the model file
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Load the model
model = load_model()

# Custom CSS for sidebar styling
st.markdown(
    """
    <style>
    /* Sidebar container styling */
    .stSidebar {
        background-color: #2f3841;
        color: white;
    }
    
    /* Styling for sidebar text and labels */
    .stSidebar .st-cqfsce,
    .stSidebar .st-emotion-cache-1gwvy71,
    .stSidebar .st-emotion-cache-6qob1r,
    .stSidebar .st-emotion-cache-1mi2ry5 {
        color: white;
    }

    /* Styling for sidebar input fields */
    .stSidebar input,
    .stSidebar select,
    .stSidebar .st-emotion-cache-1ny7cjd {
        color: white;
        background-color: #2f3841;
        border: 1px solid white;
    }

    .stSidebar .st-emotion-cache-1ny7cjd:hover {
        background-color: grey;
        border-color: grey;
    }

    /* Ensure sidebar labels are visible and white */
    .stSidebar label {
        color: white !important;
    }

    .stSidebar .st-emotion-cache-1mi2ry5 {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS to style the sidebar header
st.markdown(
    """
    <style>
    /* Target the sidebar header */
    .st-emotion-cache-uzeiqp h2 {
        color: #30373e; /* Update the color to #30373e */
        font-size: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Enter Feature Values")

# Create input fields for each feature with proper labels
price = st.sidebar.number_input("Price", min_value=0.0, value=19.99, step=0.01)
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
sales_rank_current = st.sidebar.number_input("Sales Rank: Current", min_value=0, value=500)
reviews_rating = st.sidebar.slider("Reviews: Rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
reviews_review_count = st.sidebar.number_input("Reviews: Review Count", min_value=0, value=100)
buy_box_stock = st.sidebar.number_input("Buy Box 🚚: Stock", min_value=0, value=10)
buy_box_is_fba = st.sidebar.selectbox("Buy Box: Is FBA", options=["Yes", "No"], index=0)
prime_eligible = st.sidebar.selectbox("Prime Eligible (Buy Box)", options=["Yes", "No"], index=0)
subscribe_save = st.sidebar.selectbox("Subscribe and Save", options=["Yes", "No"], index=0)
price_change_month = st.sidebar.number_input("Price Change Month", min_value=1, max_value=12, value=6)
price_change_day = st.sidebar.number_input("Price Change Day", min_value=1, max_value=31, value=15)

# Add some space between the content (optional)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Display the second image below the feature inputs
st.sidebar.image(img2, use_column_width=True)

# Process the input data and make predictions
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

    # Check and update the ColumnTransformer
    input_data_transformed = model.named_steps['column_trans'].transform(input_data)

    # Make predictions
    prediction = model.named_steps['grad_model'].predict(input_data_transformed)
    prediction_proba = model.named_steps['grad_model'].predict_proba(input_data_transformed)

    # Convert prediction to meaningful label
    if prediction[0] == 1:
        prediction_result = "Buy-Box"
    else:
        prediction_result = "Not Buy-Box"
    
    # Display prediction result
    st.header("Prediction Results:")
    st.subheader(f"This Product Has **{prediction_result}**")

    # Display probabilistic value
    st.write(f"Probability of being Buy-Box: **{prediction_proba[0][1]:.2f}**")
    st.write(f"Probability of not being Buy-Box: **{prediction_proba[0][0]:.2f}**")

    # Add space of 2cm (approximately 80 pixels)
    st.markdown("<div style='margin-bottom: 80px;'></div>", unsafe_allow_html=True)

    # Display the feature importance image
    st.image(img3, caption='Feature Importance', use_column_width=True)

    # Add useful links
    st.markdown(
        """
        <h3>Useful Links For Extra Informations</h3>
        <ul>
            <li><a href="https://research.sabanciuniv.edu/id/eprint/45502/1/10480390.pdf" target="_blank">A MACHINE LEARNING APPROACH TO UNDERSTAND THE
AMAZON BUY BOX MECHANISM</a></li>
            <li><a href="https://questromworld.bu.edu/platformstrategy/wp-content/uploads/sites/49/2022/07/PlatStrat2022_paper_52.pdf"_blank">Steering in One Click: Platform Self-Preferencing in
the Amazon Buy Box*</a></li>
            <li><a href="https://mislove.org/publications/Amazon-WWW.pdf" target="_blank">An Empirical Analysis of Algorithmic Pricing
on Amazon Marketplace</a></li>
        </ul>
        """,
        unsafe_allow_html=True
    )
