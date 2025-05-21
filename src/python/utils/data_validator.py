import torch
from .logging_utils import setup_logger

def validate_data(data_dict):
    logger = setup_logger("data_validation")
    logger.info("Starting data validation")
    
    inputs = data_dict["inputs"]
    targets = data_dict["targets"]
    
    # Check for NaN or invalid values
    if torch.isnan(inputs).any() or torch.isnan(targets).any():
        logger.error("NaN values detected in data")
        return False
    
    # Placeholder: Additional validation
    if (inputs < 0).any():
        logger.error("Negative values detected in inputs")
        return False
    
    logger.info("Data validation passed")
    return True
