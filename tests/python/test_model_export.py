import unittest
import os
from src.python.utils.model_export import export_to_onnx
from src.python.models.nexora_model import NexoraModel
import torch

class TestModelExport(unittest.TestCase):
    def test_export_to_onnx(self):
        # Create dummy model
        model = NexoraModel(input_size=10)
        model_path = "data/output/models/test_model.pth"
        onnx_path = "data/output/models/test_model.onnx"
        torch.save(model.state_dict(), model_path)
        
        # Run export
        export_to_onnx(model_path, onnx_path)
        
        # Check ONNX file exists
        self.assertTrue(os.path.exists(onnx_path))
        
        # Cleanup
        os.remove(model_path)
        os.remove(onnx_path)

if __name__ == "__main__":
    unittest.main()
