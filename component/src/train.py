import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import json

parser = argparse.ArgumentParser()

# Add Arguments
parser.add_argument("--input_data", dest="input_data", type=str)
parser.add_argument("--output_data", dest="output_data", type=str)

args = parser.parse_args()
# load the prepared csv file
prep_data = pd.read_csv(os.path.join(args.input_data, 'prep_data.csv'))

X = prep_data[['age', 'fare']]  # Adjust features based on your dataset
y = prep_data['survived']  # Assuming 'Survived' is the target variable, modify accordingly

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Store model output and scores in a dictionary
output_dict = {
    "model_coefficients": model.coef_.tolist(),
    "model_intercept": model.intercept_,
    "mean_squared_error": mse,
    "r2_score": r2
}

# Create the output folder if it doesn't exist
os.makedirs(args.output_data, exist_ok=True)

# Save the model and scores to a single JSON file within the specified folder
output_file_path = os.path.join(args.output_data, "outputs_file.json")
with open(output_file_path, 'w') as output_file:
    json.dump(output_dict, output_file)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
