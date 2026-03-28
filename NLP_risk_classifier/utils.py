import os
import pickle
from pathlib import Path

# -----------------------------------------
# Ensure directory exists
# -----------------------------------------
def ensure_dir(path):
    """
    Creates directory if it does not exist.
    Accepts both string and Path objects.
    """
    if isinstance(path, str):
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# Save pickle
# -----------------------------------------
def save_pickle(obj, path):
    """
    Save Python object to a pickle file.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# -----------------------------------------
# Load pickle
# -----------------------------------------
def load_pickle(path):
    """
    Load pickle file safely.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
