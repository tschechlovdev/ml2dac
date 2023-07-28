import ast
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration
from pandas.core.common import SettingWithCopyWarning
from smac.tae import FirstRunCrashedException

from ClusterValidityIndices.CVIHandler import CVICollection, CVI
from ClusteringCS import ClusteringCS
from Experiments import DataGeneration
from MetaLearning import LearningPhase, MetaFeatureExtractor
from MetaLearning.MetaFeatureExtractor import load_kdtree, query_kdtree, extract_meta_features
from Optimizer.OptimizerSMAC import SMACOptimizer
from Utils import Helper
from Utils.Helper import mf_set_to_string

np.random.seed(1234)

max_k_value = 100


class ApplicationPhase:
    def __init__(self, mkr_path=LearningPhase.mkr_path,
                 shape_sets=DataGeneration.generate_datasets(),  # Datasets from the Learning Phase
                 mf_set=MetaFeatureExtractor.meta_feature_sets[0],
                 k_range=None):
        self.mkr_path = mkr_path
        self.shape_sets = shape_sets
        self.mf_set = mf_set
        self.dataset_names_to_use = list(self.shape_sets.keys())
        self.k_range = k_range

    @staticmethod
    def _validate_inputs(n_warmstarts, n_loops, cvi, limit_cs):
        if n_warmstarts > n_loops:
            raise ValueError(f"n_warmstarts has to be <= n_loops, "
                             f"but we have n_warmstart={n_warmstarts} > n_loops={n_loops}")

        if cvi != "predict":
            if not isinstance(cvi, CVI):
                raise ValueError("cvi has to be either 'predict' so that we predict a CVI using meta-learning"
                                 "or it has to be an instance of our ClusterValidityIndices.CVIHandler.CVI class. "
                                 "See ClusterValidityIndices.CVIHandler.CVICollection for examples.")

        if limit_cs and n_warmstarts <= 0:
            raise ValueError("If no warmstarts are used, we cannot limit/reduce the configuration space as it depends "
                             "on the warmstarting configurations. Either use n_warmstarts > 0 or set limit_cs=False.")

    def find_similar_dataset(self, meta_features, dataset_name, n_similar_datasets=1):
        # Load kdtree --> Used to find similar dataset more efficiently
        tree = load_kdtree(path=self.mkr_path / LearningPhase.meta_feature_path, mf_set=self.mf_set)

        # Find nearest neighbors, i.e., datasets in this case
        dists, inds = query_kdtree(meta_features, tree)
        inds = inds[0]
        # We could also use the distances, but we do not need them here as the indices are already sorted by distance
        dists = dists[0]

        # Get similar datasets in their order w.r.t. distance
        most_similar_dataset_names = [self.dataset_names_to_use[ind] for ind in inds]

        if dataset_name == most_similar_dataset_names[0]:
            # In the experiments of our paper, we might have the same dataset in the MKR.
            # Therefore, we do not want to use it here and use the next-similar dataset
            D_s = most_similar_dataset_names[1:n_similar_datasets + 1]
        else:
            # Get the most-similar dataset, to use more datasets the following code has to be slightly adapted
            # However, we figured out that using more of less similar datasets leads to a performance decrease!
            D_s = most_similar_dataset_names[0:n_similar_datasets]
        return D_s

    @staticmethod
    def _remove_duplicates_from_ARI_s(ARI_s):
        ARI_s["algorithm"] = ApplicationPhase._assign_algorithm_column(ARI_s)
        ARI_s = ARI_s.drop_duplicates(subset="config", keep='first')
        ARI_s = ARI_s.drop_duplicates(subset=["algorithm", "ARI"], keep='first')
        ARI_s = ARI_s.drop("algorithm", axis=1)
        return ARI_s

    @staticmethod
    def _get_warmstart_config_from_results(warmstart_configs):
        print(warmstart_configs)
        # the configs are saved as strings, so we need ast.literal_eval to convert them to dictionaries
        warmstart_configs = [ast.literal_eval(config_string) for config_string in warmstart_configs]
        return warmstart_configs

    @staticmethod
    def _assign_algorithm_column(df_with_configs):
        return df_with_configs.apply(
            lambda x: ast.literal_eval(x["config"])["algorithm"],
            axis="columns")

    def predict_cvi(self, MF, dataset_name=None):
        # Retrieve classification model to predict CVI
        try:
            classifier_instance = joblib.load(
                f"{self.mkr_path}/models/{Helper.get_model_name()}/{mf_set_to_string(self.mf_set)}/{dataset_name}")
        except FileNotFoundError:
            # Did not use this dataset in learning phase, so use default model that we trained on all datasets
            classifier_instance = joblib.load(
                f"{self.mkr_path}/models/{Helper.get_model_name()}/{mf_set_to_string(self.mf_set)}/{None}")

        predicted_cvi = classifier_instance.predict(MF.reshape(1, -1))[0]
        predicted_cvi = CVICollection.get_cvi_by_abbrev(predicted_cvi)
        return predicted_cvi

    @staticmethod
    def select_warmstart_configurations(ARI_s, n_warmstarts):
        if n_warmstarts > 0:
            ARI_s["config_new"] = ARI_s["config"].astype('str')
            ARI_s["config_new"] = ARI_s.apply(func=lambda x: ast.literal_eval(x["config"]), axis=1)

            # Filter for configurations with less than max_k_value -> Use this for real-world data
            ARI_s = ARI_s[[isinstance(x, dict) and
                           ((not "n_clusters" in x.keys()) or (("n_clusters" in x.keys())
                                                               and (x['n_clusters'] <= max_k_value)))
                           and x["algorithm"] not in ([ClusteringCS.SPECTRAL_ALGORITHM,
                                                       ClusteringCS.AFFINITY_PROPAGATION_ALGORITHM,
                                                       ClusteringCS.MEAN_SHIFT_ALGORITHM])
                           for x in ARI_s["config_new"]]]
            ARI_s = ARI_s.drop("config_new", axis=1)

            warmstart_configs = ARI_s.sort_values("ARI", ascending=True)[0:n_warmstarts]
            print(f"Selected Warmstart Configs:")
            print(warmstart_configs["config"])
            print("--")
            return warmstart_configs
        else:
            return []
    
    def define_config_space(self, warmstart_configs, limit_cs=True):
        if limit_cs:
            warmstart_configs["algorithm"] = ApplicationPhase._assign_algorithm_column(warmstart_configs)
            algorithms = list(warmstart_configs["algorithm"].unique())
            # Use algorithms from warmstarts to build CS
            cs = ClusteringCS.build_config_space(clustering_algorithms=algorithms, k_range=self.k_range)
        else:
            algorithms = "all"
            # Use default config space
            cs = ClusteringCS.build_config_space(k_range=self.k_range)
        return cs, algorithms

    @staticmethod
    def retrieve_ARI_values_for_similar_dataset(EC, D_s):
        if not isinstance(D_s, list):
            D_s = [D_s]
        EC_s = EC[EC["dataset"].isin(D_s)]
        ARI_s = EC_s[["config", "ARI"]]
        ARI_s = ApplicationPhase._remove_duplicates_from_ARI_s(ARI_s)
        return ARI_s

    def optimize_with_meta_learning(self, X,
                                    dataset_name=None,
                                    # Can provide a name of the dataset --> comfortable for evaluation
                                    n_warmstarts=25,
                                    n_optimizer_loops=100,
                                    cvi="predict",  # Otherwise, a CVI from the CVICollection
                                    limit_cs=True,  # Used to reduce the configuration space based on warmstart configs
                                    time_limit=120 * 60,  # Set default timeout after 2 hours of optimization
                                    optimizer=SMACOptimizer,
                                    n_similar_datasets=1,
                                    seed=1234):

        print("----------------------------------")
        self._validate_inputs(n_warmstarts, n_optimizer_loops, cvi, limit_cs)

        # keeps track of additional information, e.g., mf extraction time, selected cvi, selected algorithms, etc.
        additional_result_info = {"dataset": dataset_name}
        # retrieve evaluated configurations from mkr
        EC = pd.read_csv(self.mkr_path / LearningPhase.evaluated_configs_filename, index_col=0)

        ### (a1) find similar dataset ###
        t0 = time.time()
        names, mf = extract_meta_features(X, self.mf_set)

        # track runtime of meta-feature extraction
        mf_time = time.time() - t0
        additional_result_info["mf time"] = mf_time

        # retrieve similar dataset
        d_s = self.find_similar_dataset(mf, dataset_name, n_similar_datasets)
        additional_result_info["similar dataset"] = d_s

        print(f"most similar dataset is: {d_s}")
        print("--")
        # retrieve evaluated configurations with ari values for d_s
        ARI_s = self.retrieve_ARI_values_for_similar_dataset(EC, d_s)

        ### (a2) select cluster validity index ###
        if cvi == "predict":
            # get classification model from mkr and use meta-features to predict a cvi
            cvi = self.predict_cvi(mf, dataset_name=dataset_name)
        additional_result_info["cvi"] = cvi.get_abbrev()
        print(f"selected cvi: {cvi.name} ({cvi.get_abbrev()})")
        print("--")

        ### (a3) select warmstart configurations ###
        warmstart_configs = self.select_warmstart_configurations(ARI_s, n_warmstarts)

        ### (a4) definition of configurations space (dependent on warmstart configurations) ###
        cs, algorithms = self.define_config_space(warmstart_configs, limit_cs)

        if n_warmstarts > 0:
            # update warmstart configurations
            warmstart_configs = warmstart_configs["config"]
            warmstart_configs = [ast.literal_eval(config_string) for config_string in warmstart_configs]
            warmstart_configs = [Configuration(cs, config_dict) for config_dict in warmstart_configs]

        additional_result_info["algorithms"] = algorithms
        print(f"selected algorithms: {algorithms}")
        print("--")

        ### (a5) optimizer loop ###
        print("----------------------------------")
        print("starting the optimization")
        print(cs)
        opt_instance = optimizer(dataset=X,
                                 true_labels=None,  # we do not have access to them in the application phase
                                 cvi=cvi,
                                 n_loops=n_optimizer_loops,
                                 cs=cs,
                                 wallclock_limit=time_limit,
                                 random_state=seed
                                 )

        additional_result_info["random_seed"] = seed

        not_successful = True

        while not_successful:
            try:
                opt_instance.optimize(initial_configs=warmstart_configs)
                not_successful = False
            except FirstRunCrashedException as e:
                print(e)
                print("Trying again with one less warmstart")
                warmstart_configs = warmstart_configs[1:]
                opt_instance.optimize(initial_configs=warmstart_configs)
                not_successful = False

        print("----------------------------------")
        print("finished optimization")
        print(f"best obtained configuration is:")
        print(opt_instance.get_incumbent())
        return opt_instance, additional_result_info


if __name__ == '__main__':
    warnings.filterwarnings(category=RuntimeWarning, action="ignore")
    warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")

    # define random seed
    np.random.seed(1234)
    from sklearn.datasets import make_blobs

    X, _ = make_blobs()
    ml2dac = ApplicationPhase()

    ml2dac.optimize_with_meta_learning(X=X, cvi="predict",
                                       n_warmstarts=50,
                                       n_optimizer_loops=50, limit_cs=True,
                                       n_similar_datasets=10, seed=2)
