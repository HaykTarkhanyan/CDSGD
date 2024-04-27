from typing import Union
import pandas as pd
from DSClassifierMultiQ import DSClassifierMultiQ
from sklearn.model_selection import train_test_split
from ClusteringSelector import ClusteringSelector
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.metrics import confusion_matrix, adjusted_rand_score
from scipy.stats import pearsonr


class DSClustering(DSClassifierMultiQ):
    """
    A class for clustering data using specified algorithms and parameters.
    It extends the functionalities of the DSClassifierMultiQ, tailored for
    data clustering tasks. This class provides methods to fit the model to
    data, make predictions, evaluate clustering performance, and extract key
    insights from the clustering results.
    """
    def __init__(self, data: pd.DataFrame, cluster:  Union[int, None] = None,
                 most_voted: bool = False, min_iter: int = 50,
                 max_iter: int = 400, debug_mode: bool = True,
                 lossfn: str = "MSE", num_workers: int = 0,
                 min_dloss: float = 1e-7):
        self.data = data
        self.selector = ClusteringSelector(self.data, cluster)
        self.selector.select_best_clustering()
        self.cluster_labels_df = self.selector.get_cluster_labels_df()
        if not most_voted:
            self.best = self.selector.get_best_labels()
        else:
            self.best = self.selector.get_most_voted()
        self.cluster = cluster if cluster is not None else len(set(self.best))
        self.df_with_labels = pd.concat([self.data.reset_index(drop=True),
                                         self.cluster_labels_df.reset_index(drop=True)],
                                         axis=1)
        super().__init__(self.cluster, min_iter=min_iter,
                         max_iter=max_iter, debug_mode=debug_mode,
                         lossfn=lossfn, num_workers=num_workers,
                         min_dloss=min_dloss)
        self.y_pred = None

    def generate_categorical_rules(self):
        """
        Generates rules for categorizing data points within the clusters.
        This method is used to define criteria or rules based on which data
        points are assigned to various clusters, particularly useful for
        categorical data
        """
        self.model.generate_categorical_rules(self.cluster_labels_df.values,
                                              column_names=self.cluster_labels_df.columns)

    def fit(self):
        """
        Fits the clustering model to the provided dataset.
        This method takes the training data as input and applies
        the clustering algorithm to identify patterns and group similar
        data points into clusters. It adjusts the model parameters for
        optimal clustering based on the input data.
        """
        X_train, _, y_train, _ = train_test_split(self.df_with_labels.values,
                                                  self.best,
                                                  test_size=0.6,
                                                  random_state=42)
        losses, epoch, dt = super().fit(X_train, y_train,
                                        add_single_rules=True,
                                        single_rules_breaks=3,
                                        add_mult_rules=True,
                                        column_names=self.df_with_labels.columns,
                                        print_every_epochs=31,
                                        print_final_model=False)
        return losses, epoch, dt

    def predict(self):
        """
        Predicts the cluster labels for given data points.
        Once the model is trained, this method can be used to assign cluster
        labels to new or unseen data points based on the learned clustering
        patterns. Returns a list of predicted cluster labels.
        """
        _, _, _ = self.fit()
        self.y_pred = super().predict(self.df_with_labels.values)

        return self.y_pred

    def print_most_important_rules(self, classes=None, threshold=0.2):
        """
        Prints the most significant rules or criteria used in clustering.
        This method is beneficial for understanding the key factors that
        influence the clustering process. It helps in interpreting the
        clustering model by highlighting the most influential rules.
        """
        classes = [i for i in range(self.k)]

        def check(r):
            return ("K-Means Labels" in r or "DBSCAN Labels"
                    in r or "Agglomerative Labels" in r)
        rules = self.find_most_important_rules()
        builder = ""
        for i in range(len(classes)):
            rs = rules[classes[i]]
            builder += "\n\nMost important rules for Cluster %s" % classes[i]
            for r in rs:
                if not check(r[2]):
                    builder += "\n\n\t[%.3f] R%d: %s\n\t\t" % (r[3], r[1],
                                                               r[2])
                    masses = r[4]
                    for j in range(len(masses)):
                        builder += "\t%s: %.3f" % (str(classes[j])[:3]
                                                   if j < len(classes)
                                                   else "Unc", masses[j])
        print(builder)

    def metrics(self, y=None):
        """
        Evaluates and returns the performance metrics of the clustering model.
        This method calculates various metrics such as accuracy, F1 score, and
        adjusted Rand index to assess the quality of the clusters formed.
        Useful for quantitatively evaluating the effectiveness of the
        clustering.
        """
        if y is not None:
            y = ClusteringSelector.normalize_labels(self.y_pred, y)
            print("Information of DSClassifier")
            print("\nAccuracy: %.1f%%" %
                  (accuracy_score(y, self.y_pred) * 100.))
            print("F1 Macro: %.3f" %
                  (f1_score(y, self.y_pred, average="macro")))
            print("F1 Micro: %.3f" %
                  (f1_score(y, self.y_pred, average="micro")))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y, self.y_pred))
            print("------------------")
            print("Clustering Metrics")
            rand_index = adjusted_rand_score(y, self.y_pred)
            pearson_corr, _ = pearsonr(self.y_pred, y)
            print("Rand_index: ", rand_index)
            print("Pearson: ", pearson_corr)
        print("------------------------------------------------")
        print("Silhoutte: ", (silhouette_score(self.data, self.y_pred)
                              if len(set(self.y_pred)) > 2 else 0))

    def predict_explain(self, x):
        """
        This method explain the rules of a specific instance of the data
        """
        pred, cls, rls, builder = super().predict_explain(x)
        builder = builder.replace("Class", "Cluster")
        rls = rls[~rls['rule'].str.contains('K-Means Labels|DBSCAN Labels|' +
                                            'Agglomerative Labels',
                                            case=False, regex=True)]
        return pred, cls, rls, builder
