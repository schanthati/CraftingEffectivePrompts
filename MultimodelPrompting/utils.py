

from transformers import pipeline
def query_agent(data):

    image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

    txt = image_to_text(data)
    return txt
