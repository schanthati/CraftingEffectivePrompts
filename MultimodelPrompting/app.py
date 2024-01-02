import streamlit as st
from dotenv import load_dotenv
from utils import query_agent

load_dotenv()

st.title("Image Analysis")
st.header("Please provide url of the image:")

query = st.text_area("Enter the url of an image")
print(query)
button = st.button("Generate Description of the image")

if button:
    # Get Response
    answer =  query_agent(query)
    st.write(answer)