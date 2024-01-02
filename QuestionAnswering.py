import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

  
#Function to get the response back
def getLLMResponse(context, question):

    if not question:
        raise ValueError("Question cannot be empty.") 
        #return "Question cannot be empty."
    if not context:
        raise ValueError("Context cannot be empty.") 
        #return "Context cannot be empty."
    
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})
    
    
    # Wrapper for Llama-2-7B-Chat, Running Llama 2 on CPU

    #C Transformers offers support for various open-source models, 
    #among them popular ones like Llama, GPT4All-J, MPT, and Falcon.
    #C Transformers is the Python library that provides bindings for transformer models implemented in C/C++ using the GGML library

    
    #Template for building the PROMPT
    template = """
    Given the following context:\n{context}\n\nAnswer the  question {question}.
    """

    #Creating the final PROMPT
    prompt = PromptTemplate(
    input_variables=["context","question"],
    template=template)

     #Generating the response using LLM
    response=llm(prompt.format(context=context,question=question))
    print(response)

    return response

st.set_page_config(page_title="Question Answering", page_icon=":robot_face:")
st.markdown("<h2 style='text-align: center;'>Question Answering based on the context</h2", unsafe_allow_html=True)

#Creating columns for the UI - To receive inputs from user
context = st.text_area('Enter your context', height=10)
question = st.text_area('Provide a question', height=20)

submit = st.button("Generate")

#When 'Generate' button is clicked, execute the below code
try:   
    if submit:
        st.write(getLLMResponse(context,question))
except ValueError as e:   
    st.write("Error:", str(e))
