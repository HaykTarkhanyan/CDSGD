LOWER_CONFIDENCE_BY_PROPORTION = True
OUTLIER_THRESHOLD_NUM_STD = 2

# DSClassifierMultiQ
max_iter = 500

# DBSCAN
# Initialization
eps = 0.1  # Initial epsilon value
step = 0.01  # Step size for epsilon increment
max_eps = 2  # Maximum epsilon boundary
min_samples = 5  # You can adjust this based on data density
target_clusters = 2  # Desired number of clusters excluding noise
