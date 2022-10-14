from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import _safe_indexing

##########################################################################################
# Self-implemented, adapted from sklearn implementations of Silhouette, Davies-Bould, etc.
##########################################################################################


def cop_score(X, labels):
    labels = LabelEncoder().fit(labels).transform(labels)
    n_samples = X.shape[0]
    n_labels = len(np.unique(labels))
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)
    cluster_lengths = np.zeros(n_labels)

    # main idea here is to use as separation the distance to the nearest point from the cluster
    min_inter_distances = np.zeros(n_labels)

    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        not_cluster_k = _safe_indexing(X, labels != k)

        min_inter_distance = euclidean_distances(not_cluster_k, cluster_k).min()
        min_inter_distances[k] = min_inter_distance

        cluster_lengths[k] = len(cluster_k)
        centroid = cluster_k.mean(axis=0)

        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(
            cluster_k, [centroid]))

    cop_score = np.sum((intra_dists/min_inter_distances) * cluster_lengths)

    return cop_score / n_samples


if __name__ == '__main__':
    X, y = make_blobs(n_samples=100, n_features=10)
    score = cop_score(X, y)
    print(score)

    y[0] = 10
    score = cop_score(X, y)
    print(score)
