from langchain.chains import LLMRequestsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

import os
os.environ["OPENAI_API_KEY"] = "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"

template = """
Extract the answer to the question '{query}' or say "not found" if the information is not available.
{requests_result}
"""

PROMPT = PromptTemplate(
    input_variables=["query", "requests_result"],
    template=template,
)

llm=OpenAI()

question = "Who was the first European to reach Arkansas?"
inputs = {
    "query": question,
    "url": "https://thefactfile.org/arkansas-facts/?q=" + question.replace(" ", "+"),
}

reqChain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=PROMPT))

print(reqChain(inputs))