import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

#Function to get the response back
def getLLMResponse(sentence,language):
    #llm = OpenAI(temperature=.9, model="text-davinci-003")

    # Wrapper for Llama-2-7B-Chat, Running Llama 2 on CPU

    #Quantization is reducing model precision by converting weights from 16-bit floats to 8-bit integers, 
    #enabling efficient deployment on resource-limited devices, reducing model size, and maintaining performance.

    #C Transformers offers support for various open-source models, 
    #among them popular ones like Llama, GPT4All-J, MPT, and Falcon.


    #C Transformers is the Python library that provides bindings for transformer models implemented in C/C++ using the GGML library

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})
    
    
    #Template for building the PROMPT
    template = """
    Translate this sentense {sentence} from english to {language}.
    \n\nSentence in English:
    
    """

    #Creating the final PROMPT
    prompt = PromptTemplate(
    input_variables=["sentence","language"],
    template=template)

    #Generating the response using LLM
    response=llm(prompt.format(sentence=sentence,language=language))
    print(response)

    return response

st.set_page_config(page_title="Translation from English to another language",
                    page_icon='📧',
                    layout='centered',
                    initial_sidebar_state='collapsed')
st.header("Translate now ")

form_input = st.text_area('Enter the sentence in English', height=50)

#Creating columns for the UI - To receive inputs from user
col1, col2, col3 = st.columns([10, 10, 5])
with col1:
    target_language = st.selectbox('Target Language',
                                    ('French', 'German', 'Japanese', 'Spanish',"Hindi"),
                                       index=0)


submit = st.button("Translate")

#When 'Generate' button is clicked, execute the below code
if submit:
    st.write(getLLMResponse(form_input,target_language))
