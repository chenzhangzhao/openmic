import os
DATASET_NAME: str = "openmic-2018"
SEED: int = 27
VERBOSE: int = 1
N_CLASSES: int = 20
N_SAMPLES: int = 20000
MODEL_NAME: str = "vggish"
MODELS: list = dict(vggish=dict(in_dim=1280, out_dim=128, url="https://tfhub.dev/google/vggish/1"))
AUGMENT: bool = False
TARGET_DURATION: int = 10
TARGET_SAMPLE_RATE: str = 16000
EPOCHS: int = 100
BATCH_SIZE: int =128
LEARNING_RATE: float = 1e-04
DROPOUT: int = 0.1
L2_REGULARIZATION: float = 1e-02

ROOT: str = "."
DATA_PATH: str = os.path.join(ROOT,"openmicdata\\raw",DATASET_NAME)
LABELS_MAP_PATH: str = os.path.join(DATA_PATH,"class-map.json")
AUDIO_DIR: str = os.path.join(DATA_PATH,"audio")
TARGET_AUDIO_DIR: str = (os.path.join(ROOT,"openmicdata\\processed") if not AUGMENT else os.path.join(ROOT,"openmicdata\\augmented"))
PARTITIONS_PATH: str = os.path.join(DATA_PATH,"partitions")
LABELS_PATH: str =os.path.join(DATA_PATH,"openmic-2018-aggregated-labels.csv")
EXPORT_PATH: str = os.path.join(ROOT,"output")
