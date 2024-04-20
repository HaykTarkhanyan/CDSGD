# Imports
import os
import numpy as np
import pandas as pd
from cdsgd import DSClustering

# Read the CSV
data_path = "./data/breast-cancer-wisconsin.csv"
data_path = os.path.join("..", "data", "breast-cancer-wisconsin.csv")
data = pd.read_csv(data_path)

# Replace '?' with NaN and then convert the entire DataFrame to numeric,
# coercing errors.
# This will turn all '?' to NaN.
data = data.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')

# Fill NaN values with the median of each column.
data = data.fillna(data.median())

# Extract the features and the target labels from the DataFrame, assuming
# 'class' is the target.
X_custom = data.drop(columns=['id', 'class'])  # Features matrix
y_custom = data['class']  # Target variable

# Instantiate DSClustering
# Form 1 - Default instantiation with just the feature matrix
ds1 = DSClustering(X_custom)
# Form 2 - Instantiation with a parameter to consider the most voted features
ds2 = DSClustering(X_custom, most_voted=True)
# Form 3 - Instantiation with a numeric parameter
ds3 = DSClustering(X_custom, 2)

# Apply the method to generate categorical rules
ds1.generate_categorical_rules()  # Generate rules for the first instance
ds2.generate_categorical_rules()  # Generate rules for the second instance
ds3.generate_categorical_rules()  # Generate rules for the third instance

# Apply the predict method (internally finalizes the classification model)
labels1 = ds1.predict()  # Predict labels using the first set of rules
labels2 = ds2.predict()  # Predict labels using the second set of rules
labels3 = ds3.predict()  # Predict labels using the third set of rules

# Apply the method to print the most important rules
ds1.print_most_important_rules()  # Print rules from the first model
ds2.print_most_important_rules()  # Print rules from the second model
ds3.print_most_important_rules()  # Print rules from the third model

# Apply the method to print metrics
ds1.metrics(y_custom)  # Print metrics for the first model
ds2.metrics(y_custom)  # Print metrics for the second model
ds3.metrics(y_custom)  # Print metrics for the third model
