"""
Contains the code that is responsible for extracting the meta-features for clustering.
It basically uses the pymfe package and extracts all meta-features that do not require class labels.
"""
import logging
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.cluster import MeanShift
from sklearn.neighbors._kd_tree import KDTree

from ClusterValidityIndices.CVIHandler import CVICollection
from Utils.Helper import mf_set_to_string, hopkins

logging.basicConfig(filename='metafeatures.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

pymfe_mfs = [
    # Pymfe
    ["statistical", "info-theory"],
    "general",
    "statistical",
    "info-theory",
    ["statistical", "info-theory", "general"],
    ["statistical", "general"],
    ["info-theory", "general"],
]

landmarking_mfs =[
   # AutoClust: Meanshift
    "autoclust",
]

autocluster_mfs = [
    "autocluster"
]

meta_feature_sets = pymfe_mfs + landmarking_mfs + autocluster_mfs


def extract_pymfe(X, mf_set="general"):
    mfe = MFE(groups=mf_set)
    mfe.fit(X)
    ft = mfe.extract()
    return ft[0], ft[1]


def extract_landmarking(X, mf_set):
    if mf_set == "meanshift":
        algo_instance = MeanShift()
    else:
        logging.error(f"Unknown mf set: {mf_set} - Using meanshift as default")
        algo_instance = MeanShift()

    labels = algo_instance.fit_predict(X)
    scores = []
    names = []
    for metric in CVICollection.internal_cvis:
        if len(np.unique(labels)) == 1:
            # score does not exist --> should be handled later on with a 0
            score = np.nan
        else:
            score = metric.score_cvi(X, labels)
        scores.append(score)
        names.append(metric.get_abbrev())
    return names, scores


def extract_meta_features(dataset, mf_set):
    if mf_set in pymfe_mfs:
        names, metafeatures = extract_pymfe(dataset, mf_set)
    elif mf_set in landmarking_mfs:
        names, metafeatures = extract_landmarking(dataset, mf_set)
    elif mf_set == "autocluster":
        names, metafeatures = extract_autocluster_mfes(dataset)
    else:
        logging.error(f"Unknown mf set: {mf_set} - Return None")
        print(f"Unknown mf set: {mf_set} - Return None")

        return None

    # if there are still nan values we simply set them to 0
    metafeatures = np.nan_to_num(metafeatures)
    return names, metafeatures


def extract_all_datasets(datasets, path=Path(''), mf_Set='statistical', d_names=[], save_metafeatures=True):
    default_mfe_mfs = []
    t0 = time.time()

    # contains a list of dictionary. Each dictionary contains the dataset and the corresponding meta-feature
    # names as keys and the values as value
    meta_features_per_dataset = []

    for X, d_name in zip(datasets, d_names):
        print(f"extracting metafeatures {mf_Set} from dataset {d_name}")
        names, scores = extract_meta_features(X, mf_Set)
        dic = dict(zip(names, scores))
        dic['dataset'] = d_name
        meta_features_per_dataset.append(dic)

        default_mfe_mfs.append(scores)

    if save_metafeatures:
        df = pd.DataFrame(meta_features_per_dataset)
        path.mkdir(exist_ok=True)
        df.to_csv(path / f'{mf_set_to_string(mf_Set)}_metafeatures.csv')
    default_mfe_mfs = np.array(default_mfe_mfs)

    # if there are nan values we simply set them to 0
    default_mfe_mfs = np.nan_to_num(default_mfe_mfs)
    kd_tree = KDTree(default_mfe_mfs)

    with open(path / f'{mf_set_to_string(mf_Set)}_kdtree.pkl', 'wb') as file:
        pickle.dump(kd_tree, file)
    return default_mfe_mfs


def load_kdtree(path =Path(""), mf_set='statistical'):
    with open(f'{path / mf_set_to_string(mf_set)}_kdtree.pkl', 'rb') as file:
        return pickle.load(file)


def query_kdtree(meta_features, tree, k=1):
    meta_features = np.array(meta_features)

    # if there are still nan values we simply set them to 0
    meta_features = np.nan_to_num(meta_features)
    meta_features = meta_features.reshape(1, -1)
    dists, inds = tree.query(meta_features, k=len(tree.get_arrays()[0]))
    return dists, inds


def create_kd_tree(meta_features):
    tree = KDTree(meta_features, metric="manhattan")
    return tree


def calculate_clustering_mfs(dataset):
    from sklearn.cluster import KMeans
    km_model = KMeans()
    km_model.fit(dataset)
    compactness = km_model.inertia_

    from sklearn.cluster import AgglomerativeClustering
    ag_model = AgglomerativeClustering()
    ag_model.fit(dataset)
    n_leaves = ag_model.n_leaves_

    from sklearn.cluster import OPTICS
    opt_model = OPTICS()
    opt_model.fit(dataset)
    reachability = max(opt_model.reachability_[np.where(opt_model.reachability_ != np.inf)])
    core_distance = max(opt_model.core_distances_)
    return ["compactness", "n_leaves", "reachability", "core dist"],[compactness, n_leaves, reachability, core_distance]


def extract_autocluster_mfes(dataset):
    resulting_mfs = []
    resulting_mf_names = []

    general_names, general_mfs = extract_pymfe(dataset, "general")
    stats_names, stats_mfs = extract_pymfe(dataset, "statistical")
    hopkins_mf = hopkins(dataset, len(dataset))
    hopkins_name = "hopkins"
    clustering_names, clustering_mfs = calculate_clustering_mfs(dataset)

    resulting_mfs.extend(general_mfs)
    resulting_mfs.extend(stats_mfs)
    resulting_mfs.extend(clustering_mfs)
    resulting_mfs.append(hopkins_mf)

    resulting_mf_names.extend(general_names)
    resulting_mf_names.extend(stats_names)
    resulting_mf_names.extend(clustering_names)
    resulting_mf_names.append(hopkins_name)
    return resulting_mf_names, resulting_mfs