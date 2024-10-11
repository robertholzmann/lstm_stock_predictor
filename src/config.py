# src/config.py

import os
import torch
import logging

# Configure logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def get_device():
    """
    Returns the device (CUDA if available, otherwise CPU), with an option to force CPU through an environment variable.
    """
    force_cpu = os.getenv('FORCE_CPU', 'false').lower() == 'true'
    if force_cpu:
        logger.info("FORCE_CPU is set to True. Using CPU.")
        return torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    return device

# Initialize the device
device = get_device()
