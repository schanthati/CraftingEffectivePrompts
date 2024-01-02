
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import os

def query_agent(text):

    os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

    llm = OpenAI(temperature=0.9)
    

    # Split text into chunks
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)

    # Create the text into multiple documents
    docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result=chain.run(docs)
    return(result)