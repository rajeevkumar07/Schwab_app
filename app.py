from utils.models import get_embeddings_model, get_llm_model
from langchain_chroma import Chroma
import gradio as gr
import re
import json
import yaml
from dotenv import load_dotenv
load_dotenv()

# load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# load prompt
with open("utils/prompt.yaml", "r") as f:
    prompt = yaml.safe_load(f)

# configuration
CHROMA_PATH = config["chroma_path"]

# initiate the embeddings model
embeddings_model = get_embeddings_model(config["embeddings_model"])
# initiate the model
llm = get_llm_model(config["llm_model"], config["temperature"])

# connect to the chromadb
vector_store = Chroma(
    collection_name="news_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH, 
)

# Set up the vectorstore to be the retriever
num_results = config["num_results"]
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# call this function for every message added to the chatbot
def stream_response(message, history):
    
    # retrieve the relevant chunks based on the question asked
    docs = retriever.invoke(message)

    # add all the chunks to 'knowledge'
    knowledge = ""
    for doc in docs:
        knowledge += doc.page_content+"\n\n"

    # prepare references from metadata (unique links)
    references = []
    seen_links = set()
    for doc in docs:
        link = doc.metadata.get("link") if hasattr(doc, "metadata") else None
        title = doc.metadata.get("title") if hasattr(doc, "metadata") else None
        if link and link not in seen_links:
            seen_links.add(link)
            references.append((title or link, link))

    # detect simple greeting messages to avoid adding references
    greeting_pattern = re.compile(r"^\s*(hi|helo|hey|greetings|good (morning|afternoon|evening))[,!\.\s]*$", re.IGNORECASE)
    is_greeting = bool(greeting_pattern.match(message or ""))


    # make the call to the LLM (including prompt)
    if message is not None:

        partial_message = ""

        # format the prompt
        rag_prompt = prompt["prompt"].format(message=message, history=history, knowledge=knowledge)

        # stream the response to the Gradio App
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message

        # detect if LLM indicates information is not available
        info_not_available_pattern = re.compile(r"(requested information is not available|i don't know|no information|not available|cannot find)", re.IGNORECASE)
        is_info_not_available = bool(info_not_available_pattern.search(partial_message))

        # append references at the end (after model finishes streaming)
        if references and not is_greeting and not is_info_not_available:
            refs_text_lines = [f"- {title}: {link}" for title, link in references]
            refs_block = "\n\nReferences:\n" + "\n".join(refs_text_lines)
            partial_message += refs_block
            yield partial_message

# initiate the Gradio app
news_bot = gr.ChatInterface(stream_response, title="News Bot", examples=["What is the latest news summary about Apple?", 
                                                                         "what are the 10 most popular Netflix originals ever made?", 
                                                                         "25 Top AI Stocks That Could Boost Your Portfolio?" ], 
    textbox=gr.Textbox(placeholder="Ask anything about news...",
    container=False,
    autoscroll=True,
    scale=10),
)

# launch the Gradio app
if __name__ == "__main__":
    news_bot.launch()