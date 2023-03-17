
import json

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from datetime import date

st.set_page_config(
    page_title="Prediction",
    page_icon="💸",
)


url = 'https://crypto-j2nsa5srea-ew.a.run.app'

#params = {
    #'feature1': param1,  # 0 for Sunday, 1 for Monday, ...
    #'feature2': param2
#}
response = requests.get('https://crypto-j2nsa5srea-ew.a.run.app/predict')



st.markdown("**The Prediction :**")

if st.button("Click to get our estimation of the bitcoin price in 24h"):
    st.text(response.json())



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_pred = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_a6jfvjhr.json")

st_lottie(lottie_pred, height=200)
