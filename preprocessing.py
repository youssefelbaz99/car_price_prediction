import pandas as pd
from helper import load_data
from config import TRAIN_DATA

def clean_data(df):
    # مثال: ملء القيم المفقودة (الخطوات الفعلية تعتمد على تقريرك في الـ EDA)
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # مثال: تحويل النصوص لأرقام (Encoding)
    df = pd.get_dummies(df, drop_first=True)
    
    return df

if __name__ == "__main__":
    raw_data = load_data(TRAIN_DATA)
    cleaned_data = clean_data(raw_data)
    # يمكنك حفظ البيانات النظيفة في ملف جديد إذا أردت