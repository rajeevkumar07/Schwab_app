import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.models import get_embeddings_model
from langchain_chroma import Chroma
from utils.document_processing import document_processing
from uuid import uuid4

# import the .env file
from dotenv import load_dotenv
load_dotenv()

# load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# configuration
DATA_PATH = config["data_path"]
CHROMA_PATH = config["chroma_path"]

# initiate the embeddings model
embeddings_model = get_embeddings_model(config["embeddings_model"])

# initiate the vector store
vector_store = Chroma(
    collection_name="news_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
# process the documents
docs = document_processing(DATA_PATH)

# splitting the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["chunk_size"],
    chunk_overlap=config["chunk_overlap"],
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(docs)

# creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

# adding chunks to vector store
vector_store.add_documents(documents=chunks, ids=uuids)