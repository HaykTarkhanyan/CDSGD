# Imports
import pandas as pd

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, r'C:\Users\hayk_\OneDrive\Desktop\Thesis\CDSGD')
# sys.path.insert(2, r'C:\Users\hayk_\OneDrive\Desktop\Thesis\CASED')


from cdsgd import DSClustering

# Read the CSV
data_path = "../data/gaussian_df.csv"
data = pd.read_csv(data_path)

# Since the data is already in a suitable format, we can directly use it.
X_custom = data[['x', 'y']]  # Feature matrix with 'x' and 'y'
y_custom = data['labels']  # Target variable 'labels'

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
