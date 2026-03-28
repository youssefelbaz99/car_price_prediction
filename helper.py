import pandas as pd
import pickle
import logging
from config import LOG_FILE

# دالة لتحميل البيانات
def load_data(file_path):
    return pd.read_csv(file_path)

# دالة لحفظ النموذج في ملف pkl
def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

# دالة لتحميل النموذج
def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# دالة لتسجيل النتائج
def setup_logger():
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                        format='%(asctime)s - %(message)s')
    return logging.getLogger()