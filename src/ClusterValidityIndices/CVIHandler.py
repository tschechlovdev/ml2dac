import logging
import math

import numpy as np
from hdbscan import validity_index as dbcv_score
from sklearn import metrics

from ClusterValidityIndices import DunnIndex, CogginsJain, COP_Index

"""
Responsible for everything related to Cluster Valditiy Indices (CVIs).
It contains all cvis, the cviresult class, the collection of CVIs that are used, the CVI class itself 
and the CVIEvaluator.
"""


class MetricResult:
    """
        Class that describes the information that is saved for each cvi after calculating the cvi result for a
        given kmeans result. Is used to represent the result of the MetricEvaluator.run_cvis() method.
    """

    def __init__(self, execution_time, score, name, cvi_type):
        self.execution_time = execution_time
        self.score = score
        self.name = name
        self.cvi_type = cvi_type


class CVIType:
    EXTERNAL = "External"
    INTERNAL = "Internal"


class MetricObjective:
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class CVI:
    """
        Basic entity that describes a cvi. For each cvi there is one instance of this class which will be saved in
        the MetricCollection class.
        This class is also responsible for evaluating the cvi score in a generic way.
    """

    def __init__(self, name, score_function, cvi_type, cvi_objective=MetricObjective.MAXIMIZE, sample_size=None):
        self.name = name
        self.score_function = score_function
        self.cvi_type = cvi_type
        self.sample_size = sample_size
        self.cvi_objective = cvi_objective

    def get_abbrev(self):
        return CVICollection.CVI_ABBREVIATIONS[self.name]

    def score_cvi(self, data, labels=None, true_labels=None):

        """
        Calculates the score of a cvi for a given dataset and the corresponding class labels. If the cvi is an
        external cvi, also the true_labels have to be passed to calculate the cvi. :param data: the raw dataset
        without labels
        :param data:
        :param labels: the labels that were calculated for example by kmeans
        :param true_labels: the gold standard labels of the dataset (is needed for external cvis)
        :return: the result of the cvi calculation, which should be a float. It is the negative value of a cvi if
        the cvi should be optimized (since we want to minimize the value)
        """
        if labels is None and true_labels is None:
            logging.error("either labels or true_labels has to be set to get cvi score")
            raise Exception("either labels or true_labels has to be set to get cvi score")

        # set default score to highest possible score --> Infinity will throw exception later on
        score = 2147483647

        logging.info("Start scoring cvi {}".format(self.name))

        if (len(np.unique(labels)) == 1 or len(np.unique(labels)) == 0) and self.cvi_type == CVIType.INTERNAL:
            logging.warning("only 1 label - return default value")
            return score

        if -2 in labels and self.cvi_type == CVIType.INTERNAL:
            logging.warning("-2 in labels detected, this is an indication for cutoff. Returning default value")
            return score

        # if internal just calculate score by data and labels
        if self.cvi_type == CVIType.INTERNAL:
            score = self.score_function(data, labels)

        # if external cvi then we need the "ground truth" instead of the data
        elif self.cvi_type == CVIType.EXTERNAL:
            score = self.score_function(true_labels, labels)
        else:
            logging.error("There was an unknown cvi type which couldn't be calculated. The cvi is " + self.name)

        if math.isnan(score):
            logging.info("Scored cvi {} and value is NAN. Returning 2147483647 as value".format(self.name))
            return 2147483647

        if self.cvi_objective == MetricObjective.MAXIMIZE:
            score = -1 * score

        logging.info("Scored cvi {} and value is {}".format(self.name, score))
        return score


def dunn_score(X, labels):
    #### Taken from https://github.com/jqmviegas/jqm_cvi/blob/master/jqmcvi/base.py####
    return DunnIndex.dunn_fast(X, labels)


def density_based_score(X, labels):
    try:
        return dbcv_score(X, labels)
    except ValueError as ve:
        print(f"Error occured: {ve}")
        return -1.0


class MLPCVI(CVI):

    def __init__(self, mlp_model):
        super().__init__(name="MLPMetric", score_function=None, cvi_type=CVIType.INTERNAL,
                         cvi_objective=MetricObjective.MINIMIZE)
        self.mlp_model = mlp_model

    def score_cvi(self, data, labels=None, true_labels=None):
        print("scoring mlp cvi")
        cvi_scores = []

        # calculate all internal cvis
        for cvi in CVICollection.internal_cvis:
            cvi_score = cvi.score_cvi(data, labels)
            print(f"cvi score for {cvi.name} is: {cvi_score}")
            if math.isnan(cvi_score) or math.isinf(cvi_score):
                cvi_score = 2147483647
            cvi_scores.append(cvi_score)
        cvi_scores = np.array(cvi_scores).reshape(1, -1)

        # predict ARI score based on the internal cvi scores
        ari_score = self.mlp_model.predict(cvi_scores)

        return ari_score[0]


class CVICollection:
    """
        Contains all cvis that are used for the experiments. The cvis can be get by either calling all_cvis or
        using the get_all_cvis_sorted method.
    """

    # internal scores to maximize
    SILHOUETTE = CVI("Silhouette", metrics.silhouette_score, CVIType.INTERNAL)
    CALINSKI_HARABASZ = CVI("Calinski-Harabasz", metrics.calinski_harabasz_score, CVIType.INTERNAL)
    DUNN_INDEX = CVI("Dunn Index", dunn_score, CVIType.INTERNAL)
    DENSITY_BASED_VALIDATION = CVI("DBCV", density_based_score, CVIType.INTERNAL)
    COGGINS_JAIN_INDEX = CVI("Coggins Jain Index", CogginsJain.coggins_jain_score, CVIType.INTERNAL)

    # internal scores to maximize
    DAVIES_BOULDIN = CVI("Davies-Bouldin", metrics.davies_bouldin_score, CVIType.INTERNAL,
                         MetricObjective.MINIMIZE)
    COP_SCORE = CVI("COP", COP_Index.cop_score, CVIType.INTERNAL, MetricObjective.MINIMIZE)

    # external scores
    ADJUSTED_RAND = CVI("Adjusted Rand", metrics.adjusted_rand_score, CVIType.EXTERNAL)
    ADJUSTED_MUTUAL = CVI("Adjusted Mutual", metrics.adjusted_mutual_info_score, CVIType.EXTERNAL)
    HOMOGENEITY = CVI("Homogeneity", metrics.homogeneity_score, CVIType.EXTERNAL)
    V_MEASURE = CVI("V-measure", metrics.v_measure_score, CVIType.EXTERNAL)
    COMPLETENESS_SCORE = CVI("Completeness", metrics.completeness_score, CVIType.EXTERNAL)
    FOWLKES_MALLOWS = CVI("Folkes-Mallows", metrics.fowlkes_mallows_score, CVIType.EXTERNAL)

    # abbreviations are useful for, e.g., plots
    CVI_ABBREVIATIONS = {
        SILHOUETTE.name: "SIL",
        CALINSKI_HARABASZ.name: "CH",
        DAVIES_BOULDIN.name: "DBI",
        DUNN_INDEX.name: "DI",
        DENSITY_BASED_VALIDATION.name: "DBCV",
        COP_SCORE.name: "COP",
        COGGINS_JAIN_INDEX.name: "CJI",
        ADJUSTED_RAND.name: "ARI",
        ADJUSTED_MUTUAL.name: "AMI",
        HOMOGENEITY.name: "HG",
        V_MEASURE.name: "VM",
        COMPLETENESS_SCORE.name: "CS",
        FOWLKES_MALLOWS.name: "FM",
        "MLPMetric": "MLP"
    }
    internal_cvis = [CALINSKI_HARABASZ,
                     DAVIES_BOULDIN,
                     SILHOUETTE,
                     # added scores
                     DENSITY_BASED_VALIDATION,
                     DUNN_INDEX,
                     COGGINS_JAIN_INDEX,
                     COP_SCORE
                     ]
    external_cvis = [ADJUSTED_MUTUAL, ADJUSTED_RAND,
                     COMPLETENESS_SCORE,
                     FOWLKES_MALLOWS,
                     HOMOGENEITY, V_MEASURE]
    all_cvis = external_cvis + internal_cvis

    @staticmethod
    def get_cvi_by_abbrev(cvi_abbrev):
        for cvi in CVICollection.all_cvis:
            if CVICollection.CVI_ABBREVIATIONS[cvi.name] == cvi_abbrev:
                return cvi

    @staticmethod
    def get_all_cvis_sorted_by_name():
        """
        Returns all cvis in sorted order. This is important, if e.g. calculations were done and you want to map
        value to their corresponding name.
        :return:
        """
        CVICollection.all_cvis.sort(key=lambda x: x.name)
        return CVICollection.all_cvis

    @staticmethod
    def get_sorted_abbreviations_by_type():
        return [CVICollection.CVI_ABBREVIATIONS[cvi.name] for cvi
                in CVICollection.all_cvis]

    @staticmethod
    def get_sorted_abbreviations_internal_by_type():
        return [CVICollection.CVI_ABBREVIATIONS[cvi.name] for cvi
                in CVICollection.internal_cvis]

    @staticmethod
    def get_abrev_for_cvi(cvi_name):
        return CVICollection.CVI_ABBREVIATIONS[cvi_name]
