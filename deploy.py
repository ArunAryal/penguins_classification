import numpy as np
import streamlit as st
import joblib

st.title("Penguin Classification")

pipe=joblib.load("pipe.joblib")

#user input
island=st.text_input("Enter island")
bill_length=st.number_input("Enter bill length(mm)")
bill_depth=st.number_input("Enter bill depth(mm)")
flipper_length=st.number_input("Enter flipper length(mm)")
body_mass=st.number_input("Enter body mass(gram)")
sex=st.text_input("Enter sex")
X_user_input=np.array([island,bill_length,bill_depth,flipper_length,body_mass,sex],dtype=object).reshape(1,6)

if(st.button("Predict")):
    pred=pipe.predict(X_user_input)
    st.success(f"Predicted species is{pred}")