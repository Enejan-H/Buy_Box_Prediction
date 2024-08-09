import streamlit as st
import pickle
import pandas as pd
import joblib

st.header("CAR PRICE PREDICTION")

st.sidebar.title("Please select the features of the car.")

hp = st.sidebar.slider("What is the hp of your car?", 60, 200, step=5)
age = st.sidebar.slider("What is the age of your car?", 0, 50, step=1)
km = st.sidebar.slider("What is the km of your car?", 0, 200000, step=1000)
model = st.sidebar.selectbox("What is the model of your car?", ['A3', 'Clio', 'Astra', 'Clio', 'Corsa', 'Escape', 'Insignia'])
gear=st.sidebar.selectbox("What is the gearing type of your car", ['Automatic', 'Manual', 'Semi-automatic'])


my_dict = {
    "hp": hp,
    "age": age,
    "km": km,
    "model": model,
    "gearing_type": gear
}

df = pd.DataFrame.from_dict([my_dict])

st.subheader("Your Car Specs.")
st.dataframe(df)

columns = ['hp', 'km', 'age', 'model_A3', 'model_Astra', 'model_Clio',
       'model_Corsa', 'model_Espace', 'model_Insignia', 'gearing_type_Manual',
       'gearing_type_Semi-automatic']

df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

model = pickle.load(open("lasso_final_model", "rb"))

scaler = joblib.load('scaler')

df = scaler.transform(df)

prediction = model.predict(df)

st.success("The estimated price of your car is €{}. ".format(int(prediction[0])))

