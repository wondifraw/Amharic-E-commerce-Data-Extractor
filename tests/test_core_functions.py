import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.text_cleaner import remove_emoji
from preprocessing.tokenizer import tokenize_amharic
from utils.entity_labeler import label_entities

class TestCoreFunctions(unittest.TestCase):
    
    def test_remove_emoji(self):
        text = "ዋጋ 1000 ብር 😊"
        result = remove_emoji(text)
        self.assertEqual(result, "ዋጋ 1000 ብር ")
    
    def test_tokenize_amharic(self):
        text = "ዋጋ 1000 ብር"
        tokens = tokenize_amharic(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_label_entities(self):
        tokens = ["ዋጋ", "1000", "ብር"]
        labels = label_entities(tokens)
        self.assertEqual(len(labels), len(tokens))
        self.assertIn("B-Price", [label for _, label in labels])

if __name__ == '__main__':
    unittest.main()