import numpy as np
from sklearn.cluster import KMeans
import logging 

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

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
                   breaks=3, mult_rules=False):
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
        res_row = {"name": name, "MAF method": method, "dataset": dataset,
                    "breaks": breaks, "mult_rules": mult_rules,        
                   "accuracy": accuracy, "f1": f1, 
                    "confusion_matrix": conf_matrix, 
                    "training_time": dt, "epochs": epoch+1,"min_loss": losses[-1], 
                    "all_losses": losses, "datetime": now}
        
        res_df = pd.read_csv(save_path) if os.path.exists(save_path) else pd.DataFrame()
        res_df = pd.concat([res_df, pd.DataFrame([res_row])], ignore_index=True)
        res_df.to_csv(save_path, index=False)

def filter_by_rule(df, rule_lambda, lower_confidence_by_proportion=LOWER_CONFIDENCE_BY_PROPORTION,
                   only_plot=False, print_results=False):
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
    required_columns = ["labels_clustering", "distance_norm"]
    for column in required_columns:
        assert column in df.columns, f"{column} column not found in DataFrame"
    
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
    
    num_labels = df_rule["labels_clustering"].nunique()
    if print_results:
        logging.debug(f"Number of data points left after filtering: {len(df_rule)}")
        logging.debug(f"Number of clusters left after filtering: {num_labels}")  
    
    most_common_cluster = df_rule["labels_clustering"].mode().values[0]
    if print_results: logging.debug(f"Most common cluster: {most_common_cluster}")

    
    if df_rule["labels_clustering"].nunique() == 1:
        
        if print_results: logging.debug("All data points belong to the same cluster")
        confidence = df_rule["distance_norm"].mean()
        
        if print_results: logging.debug(f"Confidence: {1 - confidence}")
    else:
        if print_results: logging.debug("Data points belong to different clusters")
        # most common cluster        
        # confidence
        confidence = df_rule[df_rule["labels_clustering"] == most_common_cluster]["distance_norm"].mean()
        if print_results: logging.debug(f"Confidence: {1 - confidence}")
        
        if lower_confidence_by_proportion:
            # num of data points in most common cluster
            num_points = len(df_rule[df_rule["labels_clustering"] == most_common_cluster])
            # proportion of data points in most common cluster
            proportion = num_points / len(df_rule)
            
            confidence = confidence * proportion
            if print_results: logging.debug(f"Confidence after lowering based on proportion: {1 - confidence}")
    if most_common_cluster == 0:
        return 1 - confidence, confidence/2, confidence/2
    elif most_common_cluster == 1:
        return confidence/2, 1 - confidence, confidence/2
    else:
        raise ValueError("Most common cluster is not 0 or 1")


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