import json

import streamlit as st
import requests
from streamlit_lottie import st_lottie
from datetime import date





st.set_page_config(
    page_title="The team",
    page_icon="⚽️",
)

st.markdown("""

            Our team of four worked on the CPP project. You will find a brief description. But be sure that it is by speaking abour the club they support that they created a climat of trust and workable environment


            """)
st.markdown("---")

with open("cd/designing.css") as source_des :
    st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

st.image("raw_data/logo_real.jpeg", width=130,)

st.markdown(" **Alexis :** He initiated the project and was responsible of scapping the data from the API. As he describe himself, 21, Brussels, Chess, Crypto, Combat Sports, Finance and Engineering. Planning on freelancing as a Data Scientist as well as creating apps. Tempus Fugit ")

st.image("raw_data/logo_psg.jpeg", width=150,)

st.markdown("**Amaury :** He wisely joinded the project and was in charge of the backend. He works as a Data Analyst at Wave.ai, a small business that aims at revolutionizing the way professional coaching works (find out more [here](https://www.wave.ai/)). He lives in Saint-Germain-en-Laye near Paris, France, with his wife and dog Jackson.")

st.image("raw_data/logo_Arsenal.jpeg", width=150,)

st.markdown("**Aran :** Lucky us, he jumped in the project to take care of the model. He is a computer science student and plan to go into Artificial Intelligence, using his behind the table skills to make better decisions with his algorithms")

st.image("raw_data/logo_united.jpeg", width=150,)

st.markdown("**Vincent :** As the man in charge of the frontend, he chose to put the biggest club in the world at the end of this page. He is a project manager in traceability (his [compagny](https://www.productdna.com/en/)). He aim to learn, develop and use new skills to built a better and smarter humanised world !")


st.markdown("[![Foo](https://tenor.com/en-GB/view/cr7manchester-united-ronaldo-cr7-manchester-united-siiu-gif-23937536)](http://google.com.au/)")
