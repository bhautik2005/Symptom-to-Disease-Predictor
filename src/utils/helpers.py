import os
import yaml
import logging

def get_project_root() -> str:
    """Returns absolute path to the project root directory."""
    # Since this file is in src/utils/helpers.py, root is 3 levels up
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config() -> dict:
    """Loads and returns the configuration dictionary from config.yaml, resolving relative paths."""
    root = get_project_root()
    config_path = os.path.join(root, "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve all relative paths in data configuration to absolute paths relative to root
    if "data" in config:
        for key, val in config["data"].items():
            if val and not os.path.isabs(val):
                config["data"][key] = os.path.normpath(os.path.join(root, val))
                
    return config

def setup_logger(name: str) -> logging.Logger:
    """Set up and return a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def batch_predict(model, X, batch_size=2000):
    """Predicts in batches to prevent ArrayMemoryError on systems with low memory/disk space."""
    import numpy as np
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(model.predict(X[i:i+batch_size]))
    return np.concatenate(preds)
