import streamlit as st
from dotenv import load_dotenv
from utils import query_agent

load_dotenv()

st.title("Summarize data")
st.header("Please upload your text file :")

# Get the text file name
file = st.file_uploader("Upload Text file",type="txt")

button = st.button("Summarize and provide result")

if button:
    if file:
        
        if file.type=='text/plain':
            from io import StringIO
            stringio=StringIO(file.getvalue().decode('utf-8'))
            read_data=stringio.read()

            # Get Response
            answer =  query_agent(read_data)
            st.write(answer)