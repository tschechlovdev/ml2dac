###  min value of all clusters from separatation and compactness (sep/comp)
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances, pairwise_distances
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils import _safe_indexing


def coggins_jain_score(X, labels):
    labels = LabelEncoder().fit(labels).transform(labels)
    n_samples, _ = X.shape
    n_labels = len(np.unique(labels))
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])), dtype=float)

    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(
            cluster_k, [centroid]))

    centroid_distances = pairwise_distances(centroids)

    # prevent distances with 0 (they are the minimum but are the distance to the same centroid)
    centroid_distances[centroid_distances == 0] = np.inf


    min_centroid_distances = np.zeros(n_labels)
    # get minimum centroid distance for each cluster
    for i, centroid_distance in enumerate(centroid_distances):
        min_centroid_distance = min(centroid_distance)
        min_centroid_distances[i] = min_centroid_distance

    # calculate min coefficient of separation and compactness
    return np.min(min_centroid_distances/ intra_dists)


if __name__ == "__main__":
     X, y = make_blobs(n_samples=1000, n_features=10)
     score = coggins_jain_score(X, y)
     print(score)



