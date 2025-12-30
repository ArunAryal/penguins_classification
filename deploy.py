import streamlit as st
import joblib

st.title("Penguin Classification")

model=joblib.load("model.joblib")

bill_length=st.number_input("Enter bill length(mm)")
bill_depth=st.number_input("Enter bill depth(mm)")
flipper_length=st.number_input("Enter flipper length(mm)")
body_mass=st.number_input("Enter body mass(gram)")
sex=st.text_input("Enter sex")
sex_integer=1 if sex.lower()=='male' else 0

if(st.button("Predict")):
    pred=model.predict([[bill_length,bill_depth,flipper_length,body_mass,sex_integer]])
    st.success(f"Predicted species is{pred}")