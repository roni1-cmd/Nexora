import unittest
import torch
from src.python.models.nexora_model import NexoraModel

class TestNexoraModel(unittest.TestCase):
    def test_forward(self):
        model = NexoraModel(input_size=10, hidden_size=64, output_size=1)
        input_data = torch.randn(1, 10)
        output = model(input_data)
        self.assertEqual(output.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
