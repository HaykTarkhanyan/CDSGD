# Imports
import pandas as pd
from cdsgd import DSClustering
# Read the CSV
data_path = "../data/iris.csv"
data = pd.read_csv(data_path)

# Extract the feature matrix and the target variable
X_custom = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_custom = data['species']

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
# is needed to encode y_custom to be use in this method
ds1.metrics()  # Print metrics for the first model
ds2.metrics()  # Print metrics for the second model
ds3.metrics()  # Print metrics for the third model
