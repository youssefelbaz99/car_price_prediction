import os

DATA_DIR = "data"
LOG_DIR = "log"

TRAIN_DATA = os.path.join(DATA_DIR, "train.csv")
TEST_DATA = os.path.join(DATA_DIR, "test.csv")
MODEL_PATH = "model.pkl"
LOG_FILE = os.path.join(LOG_DIR, "run.log")