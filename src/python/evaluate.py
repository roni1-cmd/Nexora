import torch
import json
from .models.nexora_model import NexoraModel
from .utils.data_utils import load_data
from .utils.logging_utils import setup_logger

def evaluate():
    logger = setup_logger("evaluation")
    logger.info("Starting evaluation process")
    
    # Load model
    model = NexoraModel()
    model.load_state_dict(torch.load("data/output/models/nexora_model.pth"))
    model.eval()
    
    # Load test data
    test_data = load_data()
    
    # Evaluate
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        predictions = model(test_data["inputs"])
        loss = criterion(predictions, test_data["targets"]).item()
    
    # Save metrics
    metrics = {"mse_loss": loss}
    with open("data/output/metrics/evaluation.json", "w") as f:
        json.dump(metrics, f)
    logger.info(f"Evaluation complete. MSE Loss: {loss}")
