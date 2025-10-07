from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# initiate the embeddings model
def get_embeddings_model(model_name):
    return OpenAIEmbeddings(model=model_name)

def get_llm_model(model_name, temperature):
    return ChatOpenAI(temperature=temperature, model=model_name)