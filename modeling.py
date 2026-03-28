import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from preprocessing import clean_data # Import the cleaning function from our preprocessing script
# 1. Load the raw data and clean it using our function
print("Loading and cleaning data...")
raw_df = pd.read_csv('data/train.csv')
df = clean_data(raw_df)

# 2. Split data into Features (X) and Target (y)
# Target is the 'Price' column. 
# We drop 'Price' from X. We also drop 'ID' because it's just a serial number and doesn't affect the car price.
X = df.drop(['Price', 'ID'], axis=1)
y = df['Price']

# 3. Split data into Training and Validation sets
# 80% for training, 20% for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and train the machine learning model
print("Training the model... (This might take a few seconds )")
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 5. Test the model on the validation data and check accuracy
predictions = model.predict(X_val)
accuracy = r2_score(y_val, predictions)

print("\n--- Model Results ---")
print(f"Accuracy (R2 Score): {accuracy * 100:.2f}%")

# 6. Save the trained model to a .pkl file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\nModel saved successfully as 'model.pkl' ")