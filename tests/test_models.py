import os
import sys
from dotenv import load_dotenv
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Import necessary modules
import unittest
from utils.models import get_embeddings_model, get_llm_model

class TestModels(unittest.TestCase):
    def test_get_embeddings_model(self):
        embeddings_model = get_embeddings_model("text-embedding-3-large")
        self.assertIsNotNone(embeddings_model)

    def test_get_llm_model(self):
        llm_model = get_llm_model("gpt-4o-mini", 0.3)
        self.assertIsNotNone(llm_model)
if __name__ == "__main__":
    unittest.main()