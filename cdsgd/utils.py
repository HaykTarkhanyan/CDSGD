import numpy as np
from sklearn.cluster import KMeans
import logging 

from sklearn.metrics import accuracy_score, f1_score, silhouette_score, calinski_harabasz_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import plotly.express as px
import datetime 
import pandas as pd
import os
from config import *

from config import LOWER_CONFIDENCE_BY_PROPORTION, OUTLIER_THRESHOLD_NUM_STD

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", 
                        datefmt="%d-%b-%y %H:%M:%S")

def get_distance(df, model, alg="kmeans", density_radius=0.5, penalty_rate=penalty_rate):
    if alg == "kmeans":
        distances = np.min(
            np.linalg.norm(df[:, np.newaxis] - model.cluster_centers_, axis=2), axis=1) 
    else:
        distances= calculate_adjusted_density(df, model.labels_, radius=density_radius, 
                                              penalty_rate=penalty_rate, 
                                              remove_outliers=False, normalize=False)
    
    return distances
    
def remove_outliers_and_normalize(df, distance_column="distance", label_column="labels"):    
    outliers = detect_outliers_z_score(df["distance"])
    df["outlier"] = df[distance_column].apply(lambda x: x in outliers)
    df_no_outliers = df[~df["outlier"]]
    
    # min max scale the df. dont use outliers
    min_val = df_no_outliers.groupby(label_column)[distance_column].apply('min')
    max_val = df_no_outliers.groupby(label_column)[distance_column].apply('max')
    
    def scale(x, label):
        return (x - min_val[label]) / (max_val[label] - min_val[label]) if max_val[label] > min_val[label] else 0
        
    dist_norm = df.apply(lambda row: scale(row[distance_column], row[label_column]), axis=1)
    # set all values greated than 1 to 1.01
    dist_norm = np.array([1.01 if d > 1 else d for d in dist_norm])

    return dist_norm


def evaluate_classifier(*, y_actual, y_clust, dataset="train", print_results=False,
                        purpose="kmeans_eval"):    
    TP = sum((y_actual == 1) & (y_clust == 1))
    TN = sum((y_actual == 0) & (y_clust == 0))
    accuracy = (TP + TN) / len(y_actual)
    if purpose == "kmeans_eval":
        if accuracy < 0.5: # swap 1's and 0's
            y_clust = np.array([1 if label == 0 else 0 for label in y_clust])

    TP = sum((y_actual == 1) & (y_clust == 1))
    TN = sum((y_actual == 0) & (y_clust == 0))
    FP = sum((y_actual == 0) & (y_clust == 1))
    FN = sum((y_actual == 1) & (y_clust == 0))

    accuracy = (TP + TN) / len(y_actual)
    f1 = TP / (TP + 0.5 * (FP + FN))
    conf_matrix = np.array([[TN, FP], [FN, TP]])
    # f1 = f1_score(y_actual, y_clust)
    # conf_matrix = confusion_matrix(y_actual, y_clust)
    
    if print_results:
        logging.debug(f"Evaluation on {dataset}")
        logging.debug(f"\tAccuracy:  {accuracy:.2f}")
        logging.debug(f"\tF1 Score: {f1:.2f}")
        logging.debug(f"\tConfusion Matrix: \n{conf_matrix}")
    
    return {"accuracy": accuracy, "f1": f1, "confusion_matrix": conf_matrix}


def evaluate_clustering(df, labels, model=None, alg="kmeans", round_digits=3, 
                        print_results=False, dataset="train"):
    silhouette = silhouette_score(df, labels).round(round_digits)
    calinski_harabasz = calinski_harabasz_score(df, labels).round(round_digits)
    
    if alg == "kmeans" and dataset=="train":
        inertia = round(model.inertia_,round_digits)
        
    
    if print_results:
        logging.debug(f"Evaluation on {dataset}")
        logging.debug(f"\t{silhouette = }")
        logging.debug(f"\t{calinski_harabasz = }")
        if alg == "kmeans" and dataset=="train":
            logging.debug(f"\t{inertia = }")

def dbscan_predict(model, X):
    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1

    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)

        if dist[shortest_dist_idx] < model.eps:
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


def calculate_adjusted_density(data, labels, radius, penalty_rate=0.5, 
                               remove_outliers=False, normalize=False):
    """
    Calculate the adjusted density of each point in the dataset based on the number of points within a specified radius.
    The density score is penalized if neighboring points belong to a different class.

    Args:
    - data (numpy array): The dataset where each row is a point in space.
    - labels (numpy array): Class labels corresponding to each data point.
    - radius (float): The radius of the ball within which to count neighboring points.
    - penalty_rate (float): Penalty multiplier for points from different classes within the radius.
    - remove_outliers (bool): Whether to remove outliers from the dataset before calculating densities.
    
    Returns:
    - densities (numpy array): An array where each element is the adjusted density of the corresponding point in 'data'.
    """
    tree = cKDTree(data)
    densities = np.zeros(data.shape[0])
    
    for i, point in enumerate(data):
        indices = tree.query_ball_point(point, r=radius)
        class_counts = np.sum(labels[indices] != labels[i])
        # Calculate density with penalty for different class points
        densities[i] = len(indices) - 1 - (class_counts * penalty_rate)

    if remove_outliers:
        outliers = detect_outliers_z_score(densities)
        densities = np.array([d for d in densities if d not in outliers])
    
    if normalize:        
        densities = (densities - densities.min()) / (densities.max() - densities.min())

    if sum(densities) == 0:
        msg = "All densities are zero. Consider changing the radius"
        logging.error(msg)
        raise ValueError(msg) 
    
    return densities

def get_kdist_plot(X=None, k=None, radius_nbrs=1.0):
    if k is None:
        k = 2 * X.shape[-1] - 1
    nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(X)

    # For each point, compute distances to its k-nearest neighbors
    distances, indices = nbrs.kneighbors(X) 
                                       
    distances = np.sort(distances, axis=0)
    distances = distances[:, k-1]

    # Plot the sorted K-nearest neighbor distance for each point in the dataset
    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.xlabel('Points/Objects in the dataset', fontsize=12)
    plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    plt.show()
    plt.close()

def run_dbscan(X_scaled, target_clusters=target_clusters, eps=eps, min_samples=min_samples, step=step, max_eps=max_eps):
    current_clusters = 0
    while current_clusters != target_clusters and eps <= max_eps:
        # DBSCAN with current eps and min_samples
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
        
        # Exclude noise labels and count unique clusters
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        current_clusters = len(unique_labels)
        
        # Check if we have the desired number of clusters
        if current_clusters == target_clusters:
            print(f"Found the desired number of clusters: {current_clusters} at eps={eps}")
            break
        else:
            eps += step

    # If the loop completes without breaking
    if current_clusters != target_clusters:
        print("Could not find the exact number of desired clusters with the given parameters.")
    else:
        return db

def detect_outliers_z_score(data, threshold=OUTLIER_THRESHOLD_NUM_STD):
    outliers = []
    mean = np.mean(data)
    std_dev = np.std(data)
    
    for i in data:
        z_score = (i - mean) / std_dev 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers

def report_results(y_test, y_pred, epoch=None, dt=None, losses=None, method=None, dataset=None, 
                   name=None, save_results=False, save_path=None, print_results=True, 
                   breaks=3, mult_rules=False, clustering_alg=None, label_for_dist=None):
    if epoch and dt and losses:
        if print_results:
            logging.debug(f"Training Time: {dt:.2f}s")
            logging.debug(f"Epochs: {epoch+1}")
            logging.debug(f"Min Loss: {losses[-1]:.3f}")
            plt.style.use('ggplot')
            plt.plot(list(range(epoch+1)), losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(name, fontsize=10)
            plt.savefig(os.path.join("plots", f"{name}.png"))
            plt.show()
            # save to csv


    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    if print_results:
        logging.debug(f"Accuracy:  {accuracy:.2f}")
        logging.debug(f"F1 Score: {f1:.2f}")
        logging.debug(f"Confusion Matrix: \n{conf_matrix}")
    
    if save_results:
        now = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        if save_path is None:
            save_path = f"experiments.csv"
        if name is None:
            name = "No name"
        res_row = {"datetime": now, "name": name, "MAF method": method, "dataset": dataset,
                    "breaks": breaks, "mult_rules": mult_rules,        
                   "accuracy": accuracy, "f1": f1, 
                    "confusion_matrix": conf_matrix, 
                    "training_time": dt, "epochs": epoch+1,"min_loss": losses[-1], 
                    "all_losses": losses, "clustering_alg": clustering_alg, 
                    "label_for_dist": label_for_dist}
        
        res_df = pd.read_csv(save_path) if os.path.exists(save_path) else pd.DataFrame()
        res_df = pd.concat([res_df, pd.DataFrame([res_row])], ignore_index=True)
        res_df.to_csv(save_path, index=False)

def filter_by_rule(df, rule_lambda, lower_confidence_by_proportion=LOWER_CONFIDENCE_BY_PROPORTION,
                   only_plot=False, print_results=False, label_column=LABEL_COL_FOR_DIST):
    """
    Filters a DataFrame based on a given rule lambda function and calculates the confidence score.

    Note:
        The confidence score is calculated as the average distance to the centroid of the most common cluster.
        If the data points belong to the same cluster, the confidence score is the average distance to the centroid.
        If the data points belong to different clusters, the confidence score is the average distance to the centroid of the most common cluster.
        If the confidence score is lowered by proportion, the confidence score is multiplied by the proportion of data points in the most common cluster.

        Since the closer the data points are to the centroid, the better, the confidence score is calculated as 1 - ... 
    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        rule_lambda (function): A lambda function that defines the filtering rule.
        lower_confidence_by_proportion (bool, optional): Whether to lower the confidence score by proportion. 
            Defaults to True.
        only_plot (bool, optional): Whether to to only plot the data. Defaults to False.
        
    Returns:
        tuple: A tuple containing the filtered DataFrame and the confidence score.

    Example:
        filter_by_rule(df, lambda row: row["x"]>0.5 and row["y"]>0.5)
        

    """
    # formely label column was "labels_clustering"
    required_columns = [label_column, "distance_norm"]
    for column in required_columns:
        assert column in df.columns, f"{column} column not found in DataFrame"
    assert df["distance_norm"].isna().sum() == 0, "distance_norm column contains NaN values"
    assert np.isinf(df["distance_norm"]).sum() == 0, "distance_norm column contains inf values"
        
    # example is lambda row: row["x"]>0.5 and row["y"]>0.5
    df["rule_applies"] = df.apply(rule_lambda, axis=1)
    
    df_rule = df[df["rule_applies"]]
    
    if df_rule.empty:
        # logging.info("No data points left after filtering")
        return 0, 0, 1 # full uncertainty
    
    if only_plot:
        fig = px.scatter(df, x="x", y="y", color="rule_applies")
        fig = add_centroids(fig, kmeans)
        fig.show()
        return fig 
    
    num_labels = df_rule[label_column].nunique()
    if print_results:
        logging.debug(f"Number of data points left after filtering: {len(df_rule)}")
        logging.debug(f"Number of clusters left after filtering: {num_labels}")  
    
    most_common_cluster = df_rule[label_column].mode().values[0]
    if print_results: logging.debug(f"Most common cluster: {most_common_cluster}")

    
    if df_rule[label_column].nunique() == 1:
        
        if print_results: logging.debug("All data points belong to the same cluster")
        confidence = df_rule["distance_norm"].mean()
        
        if print_results: logging.debug(f"Confidence: {1 - confidence}")
    else:
        if print_results: logging.debug("Data points belong to different clusters")
        # most common cluster        
        # confidence
        confidence = df_rule[df_rule[label_column] == most_common_cluster]["distance_norm"].mean()
        if print_results: logging.debug(f"Confidence: {1 - confidence}")
        
        if lower_confidence_by_proportion:
            # num of data points in most common cluster
            num_points = len(df_rule[df_rule[label_column] == most_common_cluster])
            if num_points == 0:
                if print_results: logging.debug("No data points in most common cluster")
                raise ValueError("No data points in most common cluster")
            # proportion of data points in most common cluster
            proportion = num_points / len(df_rule)
            
            confidence = confidence * proportion
            if print_results: logging.debug(f"Confidence after lowering based on proportion: {1 - confidence}")
    
    assert not np.isnan(confidence), "Confidence is NaN"    
        
    if most_common_cluster == 0:
        return 1 - confidence, confidence/2, confidence/2
    elif most_common_cluster == 1:
        return confidence/2, 1 - confidence, confidence/2
    else:
        raise ValueError(f"Most common cluster is {most_common_cluster}, but should be 0 or 1")


def natural_breaks(data, k=5, append_infinity=False):
    km = KMeans(n_clusters=k, max_iter=150, n_init=5)
    data = map(lambda x: [x], data)
    km.fit(data)
    breaks = []
    if append_infinity:
        breaks.append(float("-inf"))
        breaks.append(float("inf"))
    for i in range(k):
        breaks.append(max(map(lambda x: x[0], filter(lambda x: km.predict(x) == i, data))))
    breaks.sort()
    return breaks


def statistic_breaks(data, k=5, sigma_tol=1, append_infinity=False):
    mu = np.mean(data)
    sigma = np.std(data)
    lsp = np.linspace(mu - sigma * sigma_tol, mu + sigma * sigma_tol, k)
    if append_infinity:
        return [float("-inf")] + [x for x in lsp] + [float("inf")]
    else:
        return [x for x in lsp]


def is_categorical(arr, max_cat = 6):
    # if len(df.unique()) <= 10:
    #     print df.unique()
    return len(np.unique(arr[~np.isnan(arr)])) <= max_cat


def normalize(a, b, c):
    a = 0 if a < 0 else 1 if a > 1 else a
    b = 0 if b < 0 else 1 if b > 1 else b
    c = 0 if c < 0 else 1 if c > 1 else c
    n = float(a + b + c)
    return a/n, b/n, c/n


def one_hot(n, k):
    a = np.zeros((len(n), k))
    for i in range(len(n)):
        a[i, int(n[i])] = 1
    return a


def h_center(z):
    return np.exp(- z * z)


def h_right(z):
    return (1 + np.tanh(z - 1))/2


def h_left(z):
    return (1 - np.tanh(z + 1))/2