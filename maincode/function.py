import os
import time
import json
import random
import numpy as np
import tensorflow as tf
import soundfile as sf
from datetime import datetime
import config as config


def set_seeds(seed_value):
    """Set all seeds for reproducibility."""
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def load_json(path):
    """Load the data from a json file."""
    with open(path, "r") as f:
        data = json.load(f)
    return data


def export_json(data, path):
    """Export the data to a json file."""
    with open(path, "w") as f:
        json.dump(data, f)


def export_hyperparams(runid, path):
    hyperparams = {
        "runid": runid,
        "model_name": config.MODEL_NAME,
        "seed": config.SEED,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "dropout": config.DROPOUT,
        "l2_regularization": config.L2_REGULARIZATION,
    }
    export_json(hyperparams, path)


def create_uid():
    """Create a unique identifier."""
    now = datetime.now()
    unix_epoch = int(now.timestamp())
    return unix_epoch


def make_dir(path):
    """Create a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def timer(func):
    """Timer decorator."""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start} seconds")
        return result

    return wrapper


def read_audio(audio_path):
    """Read an audio file."""
    audio, sr = sf.read(audio_path, dtype="float32")
    assert sr == config.TARGET_SAMPLE_RATE
    return audio
