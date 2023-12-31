
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os


# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

def temperature_control_prompt(question, temperature=0.2):
   
    prompt = f"Explain the concept of {question}."

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',     #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
            model_type='llama',
            prompt=prompt,
            config={'max_new_tokens': 256,
            'temperature': 0.01})
    
    response=llm(prompt.format(question=question,temperature=temperature))

    return response

question = "gravity"
answer_low_temp = temperature_control_prompt(question, temperature=0.2)
answer_high_temp = temperature_control_prompt(question, temperature=0.8)

print("Answer (Low Temperature):", answer_low_temp)
print("Answer (High Temperature):", answer_high_temp)

