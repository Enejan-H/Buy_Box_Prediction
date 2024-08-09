import streamlit as st
import base64
import joblib
import pandas as pd

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

# Set the background image
set_background("1.png")





# Custom CSS for header background color only
st.markdown(
    """
    <style>
    /* Target the header container */
    .st-emotion-cache-12fmjuu {
        background-color: #ff9d3c; /* Set background color to orange */
    }
    
    /* Ensure buttons and other elements within the header are not affected */
    .st-emotion-cache-1huvf7z,
    .st-emotion-cache-w3nhqi {
        background-color: transparent; /* Ensure buttons keep their original background */
        color: inherit; /* Ensure text color inherits from its parent */
        border: none; /* Remove any border if present */
    }
    
    .st-emotion-cache-1wbqy5l span {
        color: inherit; /* Ensure button text color inherits from its parent */
    }

    .st-emotion-cache-1pbsqtx {
        fill: inherit; /* Ensure icon color inherits from its parent */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Page content

# Adjust the heading with updated CSS
st.markdown("""
    <h1 style='text-align: center; color: #30373e; font-size: 55px; margin-top: -100px; margin-bottom: 0;'>Amazon Buy Box Prediction</h1>
    """, unsafe_allow_html=True)



import streamlit as st
from PIL import Image

# Load your images
img1 = Image.open("emsbay.jpg")
img2 = Image.open("whole-basket.png")

# Display the first image at the top of the sidebar
st.sidebar.image(img1, use_column_width=True)


import streamlit as st
import pickle
import pandas as pd

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Custom CSS for sidebar styling


st.markdown(
    """
    <style>
    /* Target the sidebar container */
    .st-emotion-cache-1gv3huu {
        background-color: #2f3841;
        color: white;
    }
    
    /* Target text and other elements inside the sidebar */
    .st-emotion-cache-1gv3huu .st-emotion-cache-6qob1r,
    .st-emotion-cache-1gv3huu .st-emotion-cache-1mi2ry5,
    .st-emotion-cache-1gv3huu .st-emotion-cache-1gwvy71,
    .st-emotion-cache-1gv3huu .st-emotion-cache-1v7f65g,
    .st-emotion-cache-1gv3huu .st-emotion-cache-uzeiqp,
    .st-emotion-cache-1gv3huu .st-emotion-cache-9ycgxx {
        color: white;
    }

    /* Style file uploader and button */
    .st-emotion-cache-1gv3huu .st-emotion-cache-taue2i,
    .st-emotion-cache-1gv3huu .st-emotion-cache-1ny7cjd {
        color: white;
        background-color: #2f3841;
        border-color: white;
    }
    
    .st-emotion-cache-1gv3huu .st-emotion-cache-taue2i {
        border: 2px dashed white;
    }

    .st-emotion-cache-1gv3huu .st-emotion-cache-1ny7cjd:hover {
        background-color: grey;
        border-color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Upload CSV file for prediction
# Custom CSS to style the sidebar header
st.markdown(
    """
    <style>
    /* Target the specific h2 element for the sidebar header */
    .st-emotion-cache-uzeiqp h2 {
        color: white; /* Change text color to gray */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Add some space between the content (optional)
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# Display the second image below the "Upload CSV File" section
st.sidebar.image(img2, use_column_width=True)


if uploaded_file is not None:
    # Read the uploaded CSV file
    new_data = pd.read_csv(uploaded_file)

  
    # Process the data
    if 'Buy Box Seller' in new_data.columns:
        original = new_data['Buy Box Seller']
    else:
        original = None

    # Drop unnecessary columns and process the data
    new_data.drop(columns=['ASIN', 'open_date', 'item_is_marketplace', 'Buy Box Seller'], axis=1, inplace=True, errors='ignore')

    # Ensure 'Last Price Change' is processed
    if 'Last Price Change' in new_data.columns:
        new_data['Last Price Change'] = pd.to_datetime(new_data['Last Price Change'], errors='coerce')
        new_data['Price Change Month'] = new_data['Last Price Change'].dt.month
        new_data['Price Change Day'] = new_data['Last Price Change'].dt.day
        new_data.drop(columns=['Last Price Change'], inplace=True)

    # Transform the data
    column_trans = model.named_steps['column_trans']
    new_data_transformed = column_trans.transform(new_data)

    # Make predictions
    grad_model = model.named_steps['grad_model']
    predictions = grad_model.predict(new_data_transformed)

        # Create a DataFrame with predictions
    result = pd.DataFrame(predictions, columns=['Predictions'])
    if original is not None:
        result['Original'] = original
        result['Correct'] = result['Predictions'] == result['Original']
        correct_count = result['Correct'].sum()
        accuracy = correct_count / len(result) * 100

        st.subheader("Prediction Results")
        st.write(result)
        st.write(f"Number of correct predictions: {correct_count} out of {len(result)}")
        st.write(f"Accuracy: {accuracy:.2f} %")
    else:
        st.subheader("Prediction Results")
        st.write(result)
    
    # Display the first few rows of the uploaded data
    st.subheader("Uploaded Data Preview")
    st.write(new_data.head())
    


