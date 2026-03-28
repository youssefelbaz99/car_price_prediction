import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    # 1. Create a copy
    df_clean = df.copy()

    # 2. IMPORTANT: Remove Outliers in Price (The fix for the negative score)
    # We will keep only cars with price between 500 and 100,000
    if 'Price' in df_clean.columns:
        df_clean = df_clean[(df_clean['Price'] > 500) & (df_clean['Price'] < 100000)]

    # 3. Handle Missing Values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 4. Categorical Encoding
    le = LabelEncoder()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_clean[col] = le.fit_transform(df_clean[col])

    return df_clean

if __name__ == "__main__":
    raw_df = pd.read_csv('data/train.csv')
    cleaned_df = clean_data(raw_df)
    print("Data cleaned and outliers removed! ")