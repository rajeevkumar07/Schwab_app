import os
import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import necessary modules
import unittest
from utils.document_processing import document_processing

class TestDocumentProcessing(unittest.TestCase):
    def test_document_processing(self):
        docs = document_processing("data/stock_news.json")
        self.assertEqual(len(docs), 138)

    def test_document_processing_sample_data(self):
        docs = document_processing("data/test_data.json")
        self.assertEqual(len(docs), 4)

if __name__ == "__main__":
    unittest.main()