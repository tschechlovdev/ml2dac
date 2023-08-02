#!/usr/bin/env python
# coding: utf-8
import ast
# In[1]:


import os
import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ClusteringCS import ClusteringCS
from pandas.core.common import SettingWithCopyWarning

from ClusterValidityIndices import CVIHandler
from Experiments.RelatedWork.AutoClust import run_online_on_dataset, train_mlp_model
from MetaLearning import MetaFeatureExtractor
from MetaLearning.ApplicationPhase import ApplicationPhase
from MetaLearning.MetaFeatureExtractor import load_kdtree, query_kdtree
from Optimizer.OptimizerSMAC import SMACOptimizer
from Utils import Helper

warnings.filterwarnings(category=RuntimeWarning, action="ignore")
warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")


def process_result_to_dataframe_autocluster(optimizer_result, additional_info, y):
    selected_cvi = additional_info["cvi"]
    # The result of the application phase an optimizer instance that holds the history of executed
    # configurations with their runtime, cvi score, and so on.
    # We can also access the predicted clustering labels of each configuration to compute ARI.
    optimizer_result_df = optimizer_result.get_runhistory_df()
    for key, value in additional_info.items():
        if key == "algorithms":
            value = "+".join(value)
        optimizer_result_df[key] = value

    # optimizer_result_df = Helper.add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
    optimizer_result_df["iteration"] = [i + 1 for i in range(len(optimizer_result_df))]
    optimizer_result_df["wallclock time"] = optimizer_result_df["runtime"].cumsum()

    optimizer_result_df['CVI score'] = optimizer_result_df[selected_cvi]
    optimizer_result_df['Best CVI score'] = optimizer_result_df[selected_cvi].cummin()
    optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, y)
    optimizer_result_df['Best ARI'] = optimizer_result_df.apply(
        lambda row:
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"] == row['Best CVI score'])]["ARI"].values[0],
        axis=1)

    print(optimizer_result_df)

    # We do not need the labels in the CSV file
    optimizer_result_df = optimizer_result_df.drop(selected_cvi, axis=1)
    print(optimizer_result_df)
    return optimizer_result_df


k_range = (2, 100)


def run_autoCluster_for_dataset(X, y, d_name, related_work_path, d_names, related_work_offline_result, cvi, n_loops):
    print(f"Using dataset to query: {d_name}")
    t0 = time.time()
    names, meta_features = MetaFeatureExtractor.extract_autocluster_mfes(X)
    mf_time = time.time() - t0

    tree = load_kdtree(path=related_work_path, mf_set='autocluster')
    dists, inds = query_kdtree(meta_features, tree, k=len(d_names))
    print(f"most similar datasets are: {[d_names[ind] for ind in inds[0]]}")
    inds = inds[0]
    dists = dists[0]
    dataset_name_to_distance = {d_name: dists[ind] for ind, d_name in enumerate(d_names)}
    rw_opt_result_for_dataset = related_work_offline_result[related_work_offline_result['dataset'] != d_name]
    print(rw_opt_result_for_dataset['dataset'].unique())
    rw_opt_result_for_dataset['distance'] = [dataset_name_to_distance[dataset_name] for dataset_name
                                             in rw_opt_result_for_dataset['dataset']]

    # assign algorithm column
    rw_opt_result_for_dataset["algorithm"] = rw_opt_result_for_dataset.apply(
        lambda x: ast.literal_eval(x["config"])["algorithm"],
        axis="columns")

    # sort first for distance and then for the best ARI score
    rw_opt_result_for_dataset = rw_opt_result_for_dataset.sort_values(['distance', cvi.get_abbrev()],
                                                                      ascending=[True, False])

    best_algorithm = rw_opt_result_for_dataset['algorithm'].values[0]

    ###############################################################################
    # 2.) HPO with/for best algorithm
    # 2.1) build config space for algo
    best_algo_cs = ClusteringCS.build_paramter_space_per_algorithm(k_range=k_range)[best_algorithm]

    # 2.3) Optimize Hyperparameters with the CVI --> Actually, they would perform a grid search, but thats too time-consuming
    opt_instance = SMACOptimizer(dataset=X, true_labels=y,
                                 cvi=cvi,
                                 n_loops=n_loops, cs=best_algo_cs)
    opt_instance.optimize()

    return opt_instance


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVIHandler.CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


def process_result_to_dataframe(optimizer_result, additional_info, y):
    selected_cvi = additional_info["cvi"]
    # The result of the application phase an optimizer instance that holds the history of executed
    # configurations with their runtime, cvi score, and so on.
    # We can also access the predicted clustering labels of each configuration to compute ARI.
    optimizer_result_df = optimizer_result.get_runhistory_df()
    print(optimizer_result_df)
    for key, value in additional_info.items():
        if key == "algorithms":
            value = "+".join(value)
        if key == "similar dataset":
            value = "+".join(value)
        optimizer_result_df[key] = value

    # optimizer_result_df = Helper.add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
    optimizer_result_df["iteration"] = [i + 1 for i in range(len(optimizer_result_df))]
    optimizer_result_df["wallclock time"] = optimizer_result_df["runtime"].cumsum()

    optimizer_result_df['CVI score'] = optimizer_result_df[selected_cvi]
    optimizer_result_df['Best CVI score'] = optimizer_result_df[selected_cvi].cummin()
    optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, y)
    optimizer_result_df['Best ARI'] = optimizer_result_df.apply(
        lambda row:
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"] == row['Best CVI score'])]["ARI"].values[0],
        axis=1)

    print(optimizer_result_df)

    # We do not need the labels in the CSV file
    optimizer_result_df = optimizer_result_df.drop("labels", axis=1)
    optimizer_result_df = optimizer_result_df.drop(selected_cvi, axis=1)
    print(optimizer_result_df)
    return optimizer_result_df


def clean_up_optimizer_directory(optimizer_instance):
    if os.path.exists(optimizer_instance.output_dir) and os.path.isdir(optimizer_instance.output_dir):
        shutil.rmtree(optimizer_instance.output_dir)


def run_our_approach(X, n_loops, n_warmstarts, limit_cs, cvi, time_limit, mf_set, dataset_name, mkr_path, y,
                     random_seed):
    # Instantiate our approach
    ML2DAC = ApplicationPhase(mkr_path=mkr_path, mf_set=mf_set, k_range=k_range)

    # try:
    optimizer_instance, additional_info = ML2DAC.optimize_with_meta_learning(X,
                                                                             n_warmstarts=n_warmstarts,
                                                                             n_optimizer_loops=n_loops,
                                                                             limit_cs=limit_cs,
                                                                             cvi=cvi,
                                                                             time_limit=time_limit,
                                                                             dataset_name=dataset_name,
                                                                             n_similar_datasets=1,
                                                                             seed=random_seed)
    print(additional_info)
    optimizer_result_df = process_result_to_dataframe(optimizer_instance, additional_info, y)
    # Cleanup optimizer directory
    clean_up_optimizer_directory(optimizer_instance)
    return optimizer_result_df


def run_experiment(runs=1, run_ml2dac=True, run_baselines=True, n_warmstarts=50, n_loops=100, time_limit=120 * 60,
                   cvi="predict", limit_cs=True):
    # runs = 1
    # run_ml2dac = True

    # Do not run baselines to save time per default --> Change to False
    # run_baselines = True

    # Reduce this to get faster results, but they will probably be less accurate!
    # n_warmstarts = 50  # Number of warmstart configurations (has to be smaller than n_loops)
    # n_loops = 100  # Number of optimizer loops. This is n_loops = n_warmstarts + x
    # time_limit = 120 * 60  # Time limit of overall optimization --> Aborts earlier if n_loops not finished but time_limit reached
    # cvi = "predict"  # We want to predict a cvi based on our meta-knowledge

    # other params
    # limit_cs = True  # Reduces the search space to suitable algorithms, depending on warmstart configurations

    np.random.seed(0)

    # Specify where to find our MKR
    mkr_path = Path("src/MetaKnowledgeRepository/")
    real_world_path = "real_world_data/"
    path_to_store_results = Path("gen_results/evaluation_results/real_world/ML2DAC/")
    files = [f for f in os.listdir(real_world_path) if os.path.isfile(real_world_path + f)]
    n_similar_datasets = 1

    print("Real-world Datasets:")
    print(files)
    print(len(files))

    # Our Approach (Warmstarts)

    mf_sets_to_use = [  # MetaFeatureExtractor.meta_feature_sets[2], # "statistical"
        MetaFeatureExtractor.meta_feature_sets[4],  # ["statistical", "info-theory", "general"]
        MetaFeatureExtractor.meta_feature_sets[5],  # ["statistical", "general"]
        # MetaFeatureExtractor.meta_feature_sets[8] # "autocluster"
    ]

    print(mf_sets_to_use)

    if run_ml2dac:
        for mf_set in mf_sets_to_use:
            print("-------------")
            print(f"Running with mf_set={mf_set}")
            for run in list(range(runs)):
                seed = run * 1234
                for f in files:
                    df = pd.read_csv(real_world_path + f)
                    X = df.iloc[:, :-1]
                    y = df.iloc[:, -1]
                    print("---------------------------------")
                    print(f"Running on dataset: {f}")
                    print(f"Shape is: {X.shape}")

                    X = X.to_numpy()
                    y = y.to_numpy()
                    mf_set_string = Helper.mf_set_to_string(mf_set)
                    result_path = Path(
                        path_to_store_results) / f"run_{run}" / Path(
                        mf_set_string)

                    if not result_path.exists():
                        result_path.mkdir(exist_ok=True, parents=True)

                    optimizer_result_df = run_our_approach(X, n_loops, n_warmstarts, limit_cs,
                                                           cvi, time_limit, mf_set, f, mkr_path, y,
                                                           random_seed=seed)
                    optimizer_result_df.to_csv(result_path / f, index=False)

    if run_baselines:
        ###########################################################################################
        ###################### AutoML4Clust ##########################################################
        path_to_store_results = Path("gen_results/evaluation_results/real_world/Baselines/AML4C")
        for f in files:
            df = pd.read_csv(real_world_path + f)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            print("---------------------------------")
            print(f"Running on dataset: {f}")
            print(f"Shape is: {X.shape}")

            X = X.to_numpy()
            y = y.to_numpy()
            for cvi in [CVIHandler.CVICollection.DENSITY_BASED_VALIDATION,
                        CVIHandler.CVICollection.COP_SCORE]:

                optimizer_result_df = run_our_approach(X=X,
                                                       n_loops=n_loops,
                                                       cvi=cvi,
                                                       n_warmstarts=n_warmstarts,
                                                       limit_cs=limit_cs,
                                                       time_limit=time_limit,
                                                       dataset_name=f,
                                                       mkr_path=mkr_path,
                                                       y=y,
                                                       mf_set=MetaFeatureExtractor.meta_feature_sets[4],
                                                       random_seed=seed)
                result_path = Path(path_to_store_results) / Path(cvi.get_abbrev())

                print(f"Storing result to {result_path}")
                if not result_path.exists():
                    result_path.mkdir(exist_ok=True, parents=True)

                optimizer_result_df.to_csv(result_path / f, index=False)

                print("---------------------------------")

        ###########################################################################################
        ###################### AutoClust ##########################################################
        # define random seed

        np.random.seed(1234)

        related_work_path = Path("src/Experiments/RelatedWork/related_work")
        related_work_offline_result = pd.read_csv(related_work_path / 'related_work_offline_opt.csv',
                                                  index_col=None)
        print(related_work_offline_result.isna().sum())
        related_work_offline_result = related_work_offline_result.dropna(how="any")
        mlp_model = train_mlp_model(related_work_offline_result)

        result_path = Path("gen_results/evaluation_results/real_world/Baselines/AutoClust")

        if not result_path.exists():
            result_path.mkdir(exist_ok=True, parents=True)

        d_names = related_work_offline_result["dataset"].unique()
        for f in files:
            df = pd.read_csv(real_world_path + f)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            print("---------------------------------")
            print(f"Running on dataset: {f}")
            print(f"Shape is: {X.shape}")

            X = X.to_numpy()
            y = y.to_numpy()

            result_file = result_path / Path(f)

            autoclust_df = run_online_on_dataset(X, y, f, mlp_model, n_loops=n_loops, k_range=k_range)
            # additional_info = {"cvi": "MLP"}
            # autoclust_df = process_result_to_dataframe(autoclust_result, additional_info)
            autoclust_df.to_csv(result_file, index=False)

        #####################################################
        ########### AutoCluster #############################
        path_to_store_results = Path("gen_results/evaluation_results/real_world/Baselines/AutoCluster")
        related_work_path = Path("src/Experiments/RelatedWork/related_work")
        related_work_offline_result = pd.read_csv(related_work_path / 'related_work_offline_opt.csv',
                                                  index_col=None)
        print(related_work_offline_result.isna().sum())
        related_work_offline_result = related_work_offline_result.dropna(how="any")
        if not path_to_store_results.exists():
            path_to_store_results.mkdir(exist_ok=True, parents=True)
        d_names = related_work_offline_result["dataset"].unique()

        cvis_to_use = [CVIHandler.CVICollection.SILHOUETTE, CVIHandler.CVICollection.CALINSKI_HARABASZ,
                       CVIHandler.CVICollection.DAVIES_BOULDIN]

        for f in files:
            df = pd.read_csv(real_world_path + f)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            print("---------------------------------")
            print(f"Running on dataset: {f}")
            print(f"Shape is: {X.shape}")

            X = X.to_numpy()
            y = y.to_numpy()

            for cvi in cvis_to_use:
                result_path = path_to_store_results / Path(cvi.get_abbrev())
                if not result_path.exists():
                    result_path.mkdir(exist_ok=True, parents=True)
                result_file = result_path / Path(f)
                autocluster_result = run_autoCluster_for_dataset(X=X,
                                                                 y=y,
                                                                 d_name=f,
                                                                 related_work_path=related_work_path,
                                                                 d_names=d_names,
                                                                 related_work_offline_result=related_work_offline_result,
                                                                 cvi=cvi,
                                                                 n_loops=n_loops)
                additional_info = {"cvi": cvi.get_abbrev()}

                # autoclust_df.to_csv(result_file, index=False)
                autocluster_cvi_df = process_result_to_dataframe_autocluster(autocluster_result, additional_info, y)
                autocluster_cvi_df.to_csv(result_file, index=False)
                clean_up_optimizer_directory(autocluster_result)


if __name__ == "__main__":
    runs = 1  # 1
    run_ml2dac = True
    run_baselines = True
    n_warmstarts = 1  # 50
    n_loops = 1  # 100
    time_limit = 120 * 60
    cvi = "predict"
    limit_cs = True
    run_experiment(runs=runs, run_ml2dac=run_ml2dac, run_baselines=run_baselines, n_warmstarts=n_warmstarts,
                   n_loops=n_loops, time_limit=time_limit, cvi="predict", limit_cs=limit_cs)
