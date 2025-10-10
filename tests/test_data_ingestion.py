import unittest
import os
import json
import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
from data_ingestion import data_ingestion

class TestDataIngestion(unittest.TestCase):
    def test_data_ingestion(self):
        # Load the config file
        with open("config.json", "r") as f:
            config = json.load(f)
        data_ingestion("data/stock_news.json", "chroma_db", config)
        self.assertTrue(True)

    def test_data_ingestion_sample_data(self):
        # Load the config file
        with open("config.json", "r") as f:
            config = json.load(f)
        data_ingestion("data/test_data.json", "chroma_db", config)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()