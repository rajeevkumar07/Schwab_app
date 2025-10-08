from langchain.schema import Document
import json

class DocumentProcessing:
    """
    This class is used to process the documents.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
        return data
    
    def flatten_data(self, data):
        flattened_data = [item for _, arr in data.items() for item in arr]
        return flattened_data
    
    def create_documents(self, flattened_data):
        docs = [
            Document(
                page_content=rec["full_text"],
                metadata={"title": rec["title"], "link": rec["link"], "ticker": rec["ticker"]},
            )
            for rec in flattened_data
        ]
        return docs
# Standalone function for backward compatibility
def document_processing(data_path):
    processor = DocumentProcessing(data_path)
    data = processor.load_data()
    flattened_data = processor.flatten_data(data)
    docs = processor.create_documents(flattened_data)
    return docs