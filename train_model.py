import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np

print("Loading data...")
# Load the dataset
try:
    df = pd.read_csv("metro_data.csv")
except FileNotFoundError:
    print("Error: metro_data.csv not found. Please run data_generator.py first.")
    exit()


# Define features (X) and target (y)
features = ['station', 'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'is_peak_hour']
target = 'crowd_level'

X = df[features]
y = df[target]

# Preprocessing: One-hot encode the 'station' column since it's a categorical feature
categorical_features = ['station']
numeric_features = ['hour', 'day_of_week', 'is_weekend', 'is_holiday', 'is_peak_hour']

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model pipeline
# A pipeline bundles preprocessing and modeling steps
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model... (This may take a minute)")
# Train the model
model.fit(X_train, y_train)

print("Evaluating model...")
# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model evaluation (MSE): {mse:.4f}")
print(f"Model evaluation (RMSE): {np.sqrt(mse):.4f}")


# Save the trained model to a file using pickle
model_filename = 'crowd_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model trained and saved to {model_filename}")