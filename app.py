from utils.models import get_embeddings_model, get_llm_model
from langchain_chroma import Chroma
import gradio as gr
import re
import json
import yaml
from dotenv import load_dotenv

class NewsBot:
    def __init__(self, config_path="config.json", prompt_path="utils/prompt.yaml"):
        """Initialize the News Bot with configuration and models"""
        load_dotenv()
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.prompt = self._load_prompt(prompt_path)
        
        # Initialize models
        self.embeddings_model = get_embeddings_model(self.config["embeddings_model"])
        self.llm_model = get_llm_model(self.config["llm_model"], self.config["temperature"])
        
        # Initialize vector store
        self.vector_store = self._create_vector_store()
        self.retriever = self._create_retriever()
        
        # Initialize Gradio app
        self.app = self._create_gradio_app()
        
    def _load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _load_prompt(self, prompt_path):
        """Load prompt template from YAML file"""
        with open(prompt_path, "r") as f:
            return yaml.safe_load(f)
    
    def _create_vector_store(self):
        """Create and return the Chroma vector store"""
        return Chroma(
            collection_name="news_collection",
            embedding_function=self.embeddings_model,
            persist_directory=self.config["chroma_path"],
        )
    
    def _create_retriever(self):
        """Create and return the retriever"""
        return self.vector_store.as_retriever(
            search_kwargs={'k': self.config["num_results"]}
        )
    
    def _retrieve_documents(self, message):
        """Retrieve relevant documents for the given message"""
        return self.retriever.invoke(message)
    
    def _extract_knowledge(self, docs):
        """Extract knowledge content from retrieved documents"""
        knowledge = ""
        for doc in docs:
            knowledge += doc.page_content + "\n\n"
        return knowledge
    
    def _extract_references(self, docs):
        """Extract unique references from document metadata"""
        references = []
        seen_links = set()
        
        for doc in docs:
            link = doc.metadata.get("link") if hasattr(doc, "metadata") else None
            title = doc.metadata.get("title") if hasattr(doc, "metadata") else None
            
            if link and link not in seen_links:
                seen_links.add(link)
                references.append((title or link, link))
        
        return references
    
    def _is_greeting(self, message):
        """Check if the message is a simple greeting"""
        greeting_pattern = re.compile(
            r"^\s*(hi|helo|hey|greetings|good (morning|afternoon|evening))[,!\.\s]*$", 
            re.IGNORECASE
        )
        return bool(greeting_pattern.match(message or ""))
    
    def _is_info_not_available(self, response):
        """Check if the response indicates information is not available"""
        info_not_available_pattern = re.compile(
            r"(requested information is not available|i don't know|no information|not available|cannot find)", 
            re.IGNORECASE
        )
        return bool(info_not_available_pattern.search(response))
    
    def _format_references(self, references):
        """Format references for display"""
        refs_text_lines = [f"- {title}: {link}" for title, link in references]
        return "\n\nReferences:\n" + "\n".join(refs_text_lines)
    
    def _generate_response(self, message, history, knowledge):
        """Generate LLM response with streaming"""
        partial_message = ""
        rag_prompt = self.prompt["prompt"].format(
            message=message, 
            history=history, 
            knowledge=knowledge
        )
        
        # Stream the response
        for response in self.llm_model.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
        
        return partial_message
    
    def stream_response(self, message, history):
        """Main response streaming function for Gradio"""
        if message is None:
            return
        
        # Retrieve relevant documents
        docs = self._retrieve_documents(message)
        
        # Extract knowledge and references
        knowledge = self._extract_knowledge(docs)
        references = self._extract_references(docs)
        
        # Check message type
        is_greeting = self._is_greeting(message)
        
        # Generate response
        partial_message = ""
        for chunk in self._generate_response(message, history, knowledge):
            partial_message = chunk
            yield partial_message
        
        # Add references if appropriate
        if references and not is_greeting and not self._is_info_not_available(partial_message):
            refs_block = self._format_references(references)
            partial_message += refs_block
            yield partial_message
    
    def _create_gradio_app(self):
        """Create and configure the Gradio chat interface"""
        return gr.ChatInterface(
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
    
    def launch(self, **kwargs):
        """Launch the Gradio app"""
        self.app.launch(**kwargs)
    
    def get_app(self):
        """Get the Gradio app instance"""
        return self.app

# Standalone function for backward compatibility
def create_news_bot():
    """Create and return a NewsBot instance"""
    return NewsBot()

# Main execution
if __name__ == "__main__":
    # Create and launch the news bot
    bot = NewsBot()
    bot.launch()