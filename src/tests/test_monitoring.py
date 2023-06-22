"""
This module houses monitoring tests related to our ML model.
"""
import psutil
from ..models import train_model

def test_ram_used_for_training():
    """
    Monitoring tests: Test the amount of ram used for training in bytes does not exceed 1GB.
    """
    process = psutil.Process()
    ram_usage_before_training = process.memory_info().rss
    train_model.train()
    ram_usage_after_training = process.memory_info().rss
    ram_usage = ram_usage_after_training - ram_usage_before_training
    assert(
        ram_usage < 1000000000 # less than 1 GB
    ), f"Training the model exceeds the allowed RAM usage by {ram_usage} bytes"
    