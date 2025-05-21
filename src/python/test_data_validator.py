import unittest
import torch
from src.python.utils.data_validator import validate_data

class TestDataValidator(unittest.TestCase):
    def test_valid_data(self):
        data = {"inputs": torch.ones(10, 10), "targets": torch.ones(10, 1)}
        self.assertTrue(validate_data(data))
    
    def test_invalid_data(self):
        data = {"inputs": torch.tensor([[1.0, float('nan')]]), "targets": torch.ones(1, 1)}
        self.assertFalse(validate_data(data))

if __name__ == "__main__":
    unittest.main()
