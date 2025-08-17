import unittest
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

class TestModelPerformance(unittest.TestCase):
    
    def setUp(self):
        self.test_text = "ዋጋ 2500 ብር አድራሻ አዲስ አበባ"
        
    def test_model_inference_speed(self):
        model_name = "xlm-roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        import time
        start_time = time.time()
        
        inputs = tokenizer(self.test_text, return_tensors="pt")
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        self.assertLess(inference_time, 1.0)  # Should be under 1 second
    
    def test_batch_processing(self):
        texts = [self.test_text] * 10
        model_name = "xlm-roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        import time
        start_time = time.time()
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
        
        end_time = time.time()
        batch_time = end_time - start_time
        
        self.assertLess(batch_time, 5.0)  # Batch should complete in under 5 seconds

if __name__ == '__main__':
    unittest.main()