from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

class LLMModels:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
    def get_llm_model(self):
        """Get the LLM model"""
        return ChatOpenAI(temperature=self.temperature, model=self.model_name)

class EmbeddingsModels:
    def __init__(self, model_name):
        self.model_name = model_name
    def get_embeddings_model(self):
        """Get the embeddings model"""
        return OpenAIEmbeddings(model=self.model_name)

# initiate the embeddings model
def get_embeddings_model(model_name):
    """calling the embeddings model"""
    return EmbeddingsModels(model_name).get_embeddings_model()

def get_llm_model(model_name, temperature):
    """calling the LLM model"""
    return LLMModels(model_name, temperature).get_llm_model()