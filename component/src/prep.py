import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import os

parser = argparse.ArgumentParser()

# Add Arguments
parser.add_argument("--input_data", dest="input_data", type=str)
parser.add_argument("--output_data", dest="output_data", type=str)

# parse args
args = parser.parse_args()
# Load the titanic dataset
titanic = pd.read_csv(args.input_data)
# Drop rows with missing values in numerical columns
titanic = titanic.dropna(subset=['age', 'fare'])

# Extract numerical columns for scaling
numerical_columns = ['age', 'fare']
numerical_data = titanic[numerical_columns]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numerical data
scaled_data = scaler.fit_transform(numerical_data)

# Replace the original numerical columns with the scaled data
titanic[numerical_columns] = scaled_data
print("Hello world")
os.makedirs(Path(args.output_data), exist_ok=True)
titanic.to_csv((Path(args.output_data) / "prep_data.csv"), index=False)