import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.models import get_embeddings_model
from langchain_chroma import Chroma
from utils.document_processing import document_processing
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

class DataIngestion:
    def __init__(self, data_path, chroma_path, config):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.config = config
        self.embeddings_model = get_embeddings_model(config["embeddings_model"])
        
    def process_documents(self):
        """Process documents using the document_processing function"""
        docs = document_processing(self.data_path)
        return docs

    def create_vector_store(self):
        """Create and return the vector store"""
        vector_store = Chroma(
            collection_name="news_collection",
            embedding_function=self.embeddings_model,
            persist_directory=self.chroma_path,
        )
        return vector_store

    def create_text_splitter(self):
        """Create and return the text splitter"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"],
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter

    def create_chunks(self, docs, text_splitter):
        """Split documents into chunks"""
        chunks = text_splitter.split_documents(docs)
        return chunks

    def create_uuids(self, chunks):
        """Create unique IDs for chunks"""
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        return uuids

    def add_chunks_to_vector_store(self, chunks, uuids, vector_store):
        """Add chunks to vector store"""
        vector_store.add_documents(documents=chunks, ids=uuids)
        return vector_store


    def run_full_pipeline(self):
        """Run the complete data ingestion pipeline"""

        # Process documents
        docs = self.process_documents()
        # Create vector store
        vector_store = self.create_vector_store()
        # Create text splitter
        text_splitter = self.create_text_splitter()
        # Create chunks
        chunks = self.create_chunks(docs, text_splitter)
        # Create UUIDs
        uuids = self.create_uuids(chunks)
        # Add chunks to vector store
        vector_store = self.add_chunks_to_vector_store(chunks, uuids, vector_store)
        return vector_store

# Standalone function for backward compatibility
def data_ingestion(data_path, chroma_path, config):
    """Standalone function to run data ingestion"""
    processor = DataIngestion(data_path, chroma_path, config)
    return processor.run_full_pipeline()

# Main execution block
if __name__ == "__main__":
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Configuration
    DATA_PATH = config["data_path"]
    CHROMA_PATH = config["chroma_path"]
    
    # Create and run data ingestion
    processor = DataIngestion(DATA_PATH, CHROMA_PATH, config)
    vector_store = processor.run_full_pipeline()
    print("Data ingestion completed successfully!")