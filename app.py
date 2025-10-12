import logging
from utils.models import get_embeddings_model, get_llm_model
from langchain_chroma import Chroma
import gradio as gr
import re
import json
import yaml
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/news_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsBot:
    def __init__(self, config_path="config.json", prompt_path="utils/prompt.yaml"):
        """Initialize the News Bot with configuration and models"""
        logger.info("Initializing NewsBot")
        load_dotenv()
        
        # Load configuration
        self.config = self._load_config(config_path)
        logger.info(f"Configuration loaded successfully: {list(self.config.keys())}")
        
        self.prompt = self._load_prompt(prompt_path)
        logger.info("Prompt template loaded successfully")
        
        # Initialize models
        self.embeddings_model = get_embeddings_model(self.config["embeddings_model"])
        logger.info("Embeddings model initialized successfully")
        
        self.llm_model = get_llm_model(self.config["llm_model"], self.config["temperature"])
        logger.info("LLM model initialized successfully")
        
        # Initialize vector store
        self.vector_store = self._create_vector_store()
        logger.info("Vector store created successfully")
        
        self.retriever = self._create_retriever()
        logger.info("Retriever created successfully")
        
        # Initialize Gradio app
        self.app = self._create_gradio_app()
        logger.info("NewsBot initialization completed successfully")
        
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.debug(f"Config loaded: {config}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_prompt(self, prompt_path):
        """Load prompt template from YAML file"""
        try:
            with open(prompt_path, "r") as f:
                prompt = yaml.safe_load(f)
            logger.debug(f"Prompt loaded: {prompt}")
            return prompt
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in prompt file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading prompt: {e}")
            raise
    
    def _create_vector_store(self):
        """Create and return the Chroma vector store"""
        try:
            vector_store = Chroma(
                collection_name="news_collection",
                embedding_function=self.embeddings_model,
                persist_directory=self.config["chroma_path"],
            )
            logger.debug("Chroma vector store created successfully")
            return vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def _create_retriever(self):
        """Create and return the retriever"""
        try:
            retriever = self.vector_store.as_retriever(
                search_kwargs={'k': self.config["num_results"]}
            )
            logger.debug("Retriever created successfully")
            return retriever
        except Exception as e:
            logger.error(f"Error creating retriever: {e}")
            raise
    
    def _retrieve_documents(self, message):
        """Retrieve relevant documents for the given message"""
        try:
            docs = self.retriever.invoke(message)
            logger.info(f"Retrieved {len(docs)} documents")
            logger.debug(f"Document retrieval completed for message: {message}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
    
    def _extract_knowledge(self, docs):
        """Extract knowledge content from retrieved documents"""
        knowledge = ""
        for i, doc in enumerate(docs):
            knowledge += doc.page_content + "\n\n"
            logger.debug(f"Processed document {i+1}/{len(docs)}")
        logger.debug(f"Knowledge extraction completed, total length: {len(knowledge)} characters")
        return knowledge
    
    def _extract_references(self, docs):
        """Extract unique references from document metadata"""
        references = []
        seen_links = set()
        
        for i, doc in enumerate(docs):
            link = doc.metadata.get("link") if hasattr(doc, "metadata") else None
            title = doc.metadata.get("title") if hasattr(doc, "metadata") else None
            
            if link and link not in seen_links:
                seen_links.add(link)
                references.append((title or link, link))
                logger.debug(f"Added reference {len(references)}: {title or link}")
        
        logger.info(f"Extracted {len(references)} unique references")
        return references
    
    def _is_greeting(self, message):
        """Check if the message is a simple greeting"""
        greeting_pattern = re.compile(
            r"^\s*(hi|helo|hey|greetings|good (morning|afternoon|evening))[,!\.\s]*$", 
            re.IGNORECASE
        )
        is_greeting = bool(greeting_pattern.match(message or ""))
        logger.debug(f"Message is greeting: {is_greeting}")
        return is_greeting
    
    def _is_info_not_available(self, response):
        """Check if the response indicates information is not available"""
        info_not_available_pattern = re.compile(
            r"(requested information is not available|i don't know|no information|not available|cannot find)", 
            re.IGNORECASE
        )
        is_not_available = bool(info_not_available_pattern.search(response))
        logger.debug(f"Response indicates info not available: {is_not_available}")
        return is_not_available
    
    def _format_references(self, references):
        """Format references for display"""
        refs_text_lines = [f"- {title}: {link}" for title, link in references]
        formatted_refs = "\n\nReferences:\n" + "\n".join(refs_text_lines)
        logger.debug("References formatted successfully")
        return formatted_refs
    
    def _generate_response(self, message, history, knowledge):
        """Generate LLM response with streaming"""
        
        partial_message = ""
        rag_prompt = self.prompt["prompt"].format(
            message=message, 
            history=history, 
            knowledge=knowledge
        )
        logger.debug(f"RAG prompt length: {len(rag_prompt)} characters")
        
        # Stream the response
        chunk_count = 0
        for response in self.llm_model.stream(rag_prompt):
            partial_message += response.content
            chunk_count += 1
            yield partial_message
        
        logger.info(f"Response generation completed, total chunks: {chunk_count}, final length: {len(partial_message)}")
        return partial_message
    
    def stream_response(self, message, history):
        """Main response streaming function for Gradio"""
        
        if message is None:
            logger.warning("Received None message, returning early")
            return
        
        try:
            # Retrieve relevant documents
            docs = self._retrieve_documents(message)
            
            # Extract knowledge and references
            knowledge = self._extract_knowledge(docs)
            references = self._extract_references(docs)
            
            # Check message type
            is_greeting = self._is_greeting(message)
            logger.info(f"Message is greeting: {is_greeting}")
            
            # Generate response
            partial_message = ""
            chunk_count = 0
            for chunk in self._generate_response(message, history, knowledge):
                partial_message = chunk
                chunk_count += 1
                logger.debug(f"Streaming chunk {chunk_count}")
                yield partial_message
            
            # Add references if appropriate
            if references and not is_greeting and not self._is_info_not_available(partial_message):
                refs_block = self._format_references(references)
                partial_message += refs_block
                logger.debug("References added to response")
                yield partial_message
            else:
                logger.info("Skipping references (greeting, no info available, or no references)")
            
            logger.info(f"Stream response completed successfully, total chunks: {chunk_count}")
            
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
    
    def _create_gradio_app(self):
        """Create and configure the Gradio chat interface"""
        try:
            app = gr.ChatInterface(
                self.stream_response,
                title="News Bot",
                examples=[
                    "What is the latest news summary about Apple?",
                    "what are the 10 most popular Netflix originals ever made?",
                    "25 Top AI Stocks That Could Boost Your Portfolio?"
                ],
                textbox=gr.Textbox(
                    placeholder="Ask anything about news...",
                    container=False,
                    autoscroll=True,
                    scale=10
                ),
            )
            logger.info("Gradio app created successfully")
            return app
        except Exception as e:
            logger.error(f"Error creating Gradio app: {e}")
            raise
    
    def launch(self, **kwargs):
        """Launch the Gradio app"""
        try:
            self.app.launch(**kwargs)
            logger.info("Gradio app launched successfully")
        except Exception as e:
            logger.error(f"Error launching Gradio app: {e}")
            raise
    
    def get_app(self):
        """Get the Gradio app instance"""
        return self.app

# Standalone function for backward compatibility
def create_news_bot():
    """Create and return a NewsBot instance"""
    return NewsBot()

# Main execution
if __name__ == "__main__":
    logger.info("Starting NewsBot application")
    try:
        # Create and launch the news bot
        bot = NewsBot()
        logger.info("NewsBot created successfully, launching...")
        bot.launch()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {e}")
        raise