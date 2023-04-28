import time

import numpy as np

from ClusteringCS import ClusteringCS
from ClusterValidityIndices.CVIHandler import CVICollection, CVIType


def smac_function(config, optimizer_instance, **kwargs):
    X = optimizer_instance.dataset
    cvi = optimizer_instance.cvi
    true_labels = optimizer_instance.true_labels
    algorithm_name = config["algorithm"]

    t0 = time.time()

    clust_algo_instance = ClusteringCS.get_ALGORITHMS_MAP()[algorithm_name]
    print(f"Executing Configuration: {config}")

    # Execute clustering algorithm
    try:
        y = clust_algo_instance.execute_config(X, config)
    except ValueError as e:
        print(e)
        y = np.random(low=0, high=100, size=X.shape[0])
        
    # store additional info, such as algo and cvi runtime, and the predicted clustering labels
    algo_runtime = time.time() - t0

    cvi_start = time.time()
    # Scoring cvi, true_labels are none for internal CVI. We only use them for consistency.
    # We only want the true_labels in the learning phase, where we optimize an external CVI.
    score = cvi.score_cvi(X, labels=y, true_labels=true_labels)
    print(f"Obtained CVI score for {cvi.get_abbrev()}: {score}")
    print("----")
    cvi_runtime = time.time() - cvi_start
    add_info = {"algo_time": algo_runtime, "metric_time": cvi_runtime, "labels": y.tolist()}

    if optimizer_instance.cvi.cvi_type == CVIType.INTERNAL:
        # if we are using an internal cvi, this is the application phase
        # and we do not want to calculate all cvis additionally
        return score, add_info

    # We are in the learning phase and thus want to calculate all CVIs
    # Actually, we should do this in the LearningPhase script!
    int_cvis = CVICollection.internal_cvis
    for int_cvi in int_cvis:
        int_cvi_score = int_cvi.score_cvi(X, labels=y)
        add_info[int_cvi.get_abbrev()] = int_cvi_score

    return score, add_info
