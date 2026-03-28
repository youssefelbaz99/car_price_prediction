import pandas as pd
import pickle
from preprocessing import clean_data

# 1. Load test data
print("Loading test data...")
test_raw = pd.read_csv('data/test.csv')

# 2. Clean data 
# We call our function but we check here too
df_test = clean_data(test_raw)

# If for some reason df_test is empty, we reload it without cleaning price
if df_test.empty:
    print("Warning: Cleaned data was empty, reloading without price filter...")
    df_test = test_raw.copy()
    # Manual simple cleaning for test only
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df_test.select_dtypes(include=['object']).columns:
        df_test[col] = le.fit_transform(df_test[col].astype(str))
    df_test = df_test.fillna(0)

# 3. Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 4. Match features
model_features = model.feature_names_in_
X_test = df_test[model_features]

# 5. Predict
print(f"Predicting prices for {len(X_test)} cars...")
predictions = model.predict(X_test)

# 6. Save results
test_raw['Predicted_Price'] = predictions
test_raw.to_csv('final_predictions.csv', index=False)
print("Success! 'final_predictions.csv' is ready. ")