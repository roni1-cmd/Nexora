import torch
from src.python.models.nexora_model import NexoraModel
from src.python.utils.logging_utils import setup_logger

def export_to_onnx(model_path, onnx_path, input_size=10):
    logger = setup_logger("model_export")
    logger.info("Starting model export to ONNX")
    
    # Load model
    model = NexoraModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, input_size)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    logger.info(f"Model exported to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx("data/output/models/nexora_model.pth", "data/output/models/nexora_model.onnx")
