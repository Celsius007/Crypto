import json

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from datetime import date

st.set_page_config(
    page_title="Hello",
    page_icon="💸",
)



with open("designing.css") as source_des :
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>CPP<br>- Crypto Price Prediction -<br> Or how to get rich</h1>", unsafe_allow_html=True)

st.markdown("---")


st.markdown('''

        Welcome to our MVP.\n
        Here you will find our bitcoin price evolution prediction!
        It will give you the expected price in 24h.

        ''')

st.markdown("Beware, this is a prediction, not a certainty!")



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_json = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_27klftwl.json")

st_lottie(lottie_json, height=300)

st.markdown('''

        _This prediction is provided by : Alexis V. A., Amaury G., Aran K. & Vincent M._
        ''')
