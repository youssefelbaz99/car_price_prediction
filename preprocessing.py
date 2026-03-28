import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data(df):
    """
    This function takes the raw dataframe, cleans it, and prepares it for training.
    """
    # 1. Create a copy of the data to work on safely
    df_clean = df.copy()

    # 2. Handle Missing Values
    # Fill numerical columns with median, and categorical columns with mode
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    # 3. Categorical Encoding
    # Convert text categories (e.g., Petrol/Diesel) to numbers (0/1)
    le = LabelEncoder()
    
    # Select columns that contain text (objects)
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df_clean[col] = le.fit_transform(df_clean[col])

    # 4. Drop unnecessary columns (Optional)
    # Uncomment and add column names if there are columns like IDs that don't affect the price
    # cols_to_drop = ['car_ID'] 
    # df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

    return df_clean

# Test the preprocessing script independently
if __name__ == "__main__":
    # Load raw data
    raw_df = pd.read_csv('data/train.csv')
    
    # Apply the cleaning function
    cleaned_df = clean_data(raw_df)
    
    print("Data cleaned successfully!")
    print("\n--- Data head after encoding ---")
    print(cleaned_df.head())
    
    # Check if any missing values remain
    print("\n--- Are there any missing values left? ---")
    print(cleaned_df.isnull().sum().max() == 0) # Should print True if completely clean