import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """
    Cleans the data and only removes price outliers if the 'Price' column exists.
    """
    df_clean = df.copy()

    # 1. Remove Price Outliers ONLY if 'Price' column exists (Training phase)
    # This prevents the test data from being deleted
    if 'Price' in df_clean.columns:
        df_clean = df_clean[(df_clean['Price'] > 500) & (df_clean['Price'] < 100000)]

    # 2. Handle Missing Values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 3. Categorical Encoding
    le = LabelEncoder()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    return df_clean
if __name__ == "__main__":
    raw_df = pd.read_csv('data/train.csv')
    cleaned_df = clean_data(raw_df)
    print("Data cleaned and outliers removed! ")