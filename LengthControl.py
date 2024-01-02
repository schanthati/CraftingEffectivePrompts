
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os


def token_control_prompt(question, max_new_tokens=100):
   
    prompt = f"Explain the concept of {question}."

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',  #https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
            model_type='llama',
            prompt=prompt,
            config={'max_new_tokens': max_new_tokens,
            'temperature': 0.01})
    
    response=llm(prompt.format(question=question,max_new_tokens=max_new_tokens))

    return response

question = "gravity"
answer_max_tokens = token_control_prompt(question, max_new_tokens=250)
answer_min_tokens = token_control_prompt(question, max_new_tokens=25)

print("Answer (Max tokens):", answer_max_tokens)
print("Answer (Min tokens):", answer_min_tokens)

