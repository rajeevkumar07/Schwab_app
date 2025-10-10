import unittest
import os
import sys
# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
from app import NewsBot

class TestApp(unittest.TestCase):
    def test_app(self):
        app = NewsBot()
        self.assertIsNotNone(app)

if __name__ == "__main__":
    unittest.main()