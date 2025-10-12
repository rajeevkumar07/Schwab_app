import logging
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.models import get_embeddings_model
from langchain_chroma import Chroma
from utils.document_processing import document_processing
from uuid import uuid4
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

class DataIngestion:
    def __init__(self, data_path, chroma_path, config):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.config = config
        self.embeddings_model = get_embeddings_model(config["embeddings_model"])
        logger.info("Embeddings model initialized successfully")
        
    def process_documents(self):
        """Process documents using the document_processing function"""
        try:
            docs = document_processing(self.data_path)
            logger.info(f"Successfully processed {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

    def create_vector_store(self):
        """Create and return the vector store"""
        try:
            vector_store = Chroma(
                collection_name="news_collection",
                embedding_function=self.embeddings_model,
                persist_directory=self.chroma_path,
            )
            logger.info("Vector store created successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def create_text_splitter(self):
        """Create and return the text splitter"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"],
                length_function=len,
                is_separator_regex=False,
            )
            logger.info("Text splitter created successfully")
            return text_splitter
        except Exception as e:
            logger.error(f"Error creating text splitter: {e}")
            raise

    def create_chunks(self, docs, text_splitter):
        """Split documents into chunks"""
        try:
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise

    def create_uuids(self, chunks):
        """Create unique IDs for chunks"""
        try:
            uuids = [str(uuid4()) for _ in range(len(chunks))]
            logger.info(f"Successfully created {len(uuids)} UUIDs")
            return uuids
        except Exception as e:
            logger.error(f"Error creating UUIDs: {e}")
            raise

    def add_chunks_to_vector_store(self, chunks, uuids, vector_store):
        """Add chunks to vector store"""
        try:
            vector_store.add_documents(documents=chunks, ids=uuids)
            logger.info("Chunks added to vector store successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            raise

    def run_full_pipeline(self):
        """Run the complete data ingestion pipeline"""
        try:
            # Process documents
            logger.info("Step 1: Processing documents")
            docs = self.process_documents()
            
            # Create vector store
            logger.info("Step 2: Creating vector store")
            vector_store = self.create_vector_store()
            
            # Create text splitter
            logger.info("Step 3: Creating text splitter")
            text_splitter = self.create_text_splitter()
            
            # Create chunks
            logger.info("Step 4: Creating chunks")
            chunks = self.create_chunks(docs, text_splitter)
            
            # Create UUIDs
            logger.info("Step 5: Creating UUIDs")
            uuids = self.create_uuids(chunks)
            
            # Add chunks to vector store
            logger.info("Step 6: Adding chunks to vector store")
            vector_store = self.add_chunks_to_vector_store(chunks, uuids, vector_store)
            
            logger.info("Data ingestion pipeline completed successfully")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error in data ingestion pipeline: {e}")
            raise

# Standalone function for backward compatibility
def data_ingestion(data_path, chroma_path, config):
    """Standalone function to run data ingestion"""
    try:
        processor = DataIngestion(data_path, chroma_path, config)
        result = processor.run_full_pipeline()
        logger.info("Standalone data ingestion completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error in standalone data ingestion: {e}")
        raise

# Main execution block
if __name__ == "__main__":
    try:
        # Load configuration
        with open("config.json", "r") as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully")
        
        # Configuration
        DATA_PATH = config["data_path"]
        CHROMA_PATH = config["chroma_path"]
        logger.info(f"Using data path: {DATA_PATH}")
        logger.info(f"Using chroma path: {CHROMA_PATH}")
        
        # Create and run data ingestion
        processor = DataIngestion(DATA_PATH, CHROMA_PATH, config)
        vector_store = processor.run_full_pipeline()
        
        logger.info("Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise