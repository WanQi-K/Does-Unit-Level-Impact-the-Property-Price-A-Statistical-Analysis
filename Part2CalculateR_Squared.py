#Price Prediction
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import t

# Add 'Level' feature to the data
# Split data with and without 'Level' feature
X_with_level = InderaSubang_data[["Square Feet", "Unit Level"]]
X_without_level = InderaSubang_data[["Square Feet"]]
y = InderaSubang_data["Transaction Price"]

# Train-Test Split for both scenarios
X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(X_with_level, y, test_size=0.2, random_state=42)
X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(X_without_level, y, test_size=0.2, random_state=42)

# Train models
model_with_level = LinearRegression()
model_with_level.fit(X_train_with, y_train_with)

model_without_level = LinearRegression()
model_without_level.fit(X_train_without, y_train_without)

# Calculate R-squared
r2_with_level = model_with_level.score(X_test_with, y_test_with)
r2_without_level = model_without_level.score(X_test_without, y_test_without)

# Display R-squared values
print(f"R-squared without 'Unit Level': {r2_without_level:.4f}")
print(f"R-squared with 'Unit Level': {r2_with_level:.4f}")
