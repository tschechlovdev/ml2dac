#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from MetaLearning.ApplicationPhase import ApplicationPhase
from ClusterValidityIndices import CVIHandler
from MetaLearning import MetaFeatureExtractor
from pathlib import Path
from pandas.core.common import SettingWithCopyWarning
from Utils import Helper
import warnings
from pathlib import Path
import shutil
import time

warnings.filterwarnings(category=RuntimeWarning, action="ignore")
warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")
import numpy as np
np.random.seed(0)
# Specify where to find our MKR
mkr_path = Path("../MetaKnowledgeRepository/") 

mf_set = MetaFeatureExtractor.meta_feature_sets[2]

real_world_path = "real_world_data/"
path_to_store_results = Path("real_world_results/ML2DAC/")
files = [f for f in os.listdir(real_world_path) if os.path.isfile(real_world_path+ f)]
counter = 0
for f in files:
    print(f)
    df = pd.read_csv(real_world_path + f)
    print(df.shape)
    print("--------")
    if df.shape[0] <= 20000:
        counter+=1
print(counter)
print(len(files))


# In[3]:


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVIHandler.CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


# In[4]:


def process_result_to_dataframe(optimizer_result, additional_info):
    selected_cvi = additional_info["cvi"]
    # The result of the application phase an optimizer instance that holds the history of executed
    # configurations with their runtime, cvi score, and so on.
    # We can also access the predicted clustering labels of each configuration to compute ARI.
    optimizer_result_df = optimizer_result.get_runhistory_df()
    for key, value in additional_info.items():
        if key == "algorithms":
            value = "+".join(value)
        optimizer_result_df[key] = value

    #optimizer_result_df = Helper.add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
    optimizer_result_df["iteration"] =  [i + 1 for i in range(len(optimizer_result_df))]
    optimizer_result_df["wallclock time"] =  optimizer_result_df["runtime"].cumsum()

    optimizer_result_df['CVI score'] = optimizer_result_df[selected_cvi]
    optimizer_result_df['Best CVI score'] = optimizer_result_df[selected_cvi].cummin()
    optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, y)
    optimizer_result_df['Best ARI'] = optimizer_result_df.apply(
        lambda row: 
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"]==row['Best CVI score'])]["ARI"].values[0],
        axis=1)

    print(optimizer_result_df)

    # We do not need the labels in the CSV file
    optimizer_result_df = optimizer_result_df.drop("labels", axis=1)
    optimizer_result_df = optimizer_result_df.drop(selected_cvi, axis=1)
    print(optimizer_result_df)
    return optimizer_result_df


# In[5]:


def clean_up_optimizer_directory(optimizer_instance):
    if os.path.exists(optimizer_instance.output_dir) and os.path.isdir(optimizer_instance.output_dir):
                shutil.rmtree(optimizer_instance.output_dir)


# # Our Approach (Warmstarts)

# In[6]:


from smac.tae import FirstRunCrashedException
mf_sets_to_use = [#MetaFeatureExtractor.meta_feature_sets[2], # "statistical"
                  MetaFeatureExtractor.meta_feature_sets[4], # ["statistical", "info-theory", "general"]
                  MetaFeatureExtractor.meta_feature_sets[5], # ["statistical", "general"]
                  #MetaFeatureExtractor.meta_feature_sets[8] # "autocluster"
                 ]

print(mf_sets_to_use)

similar_datasets = [1, 2, 3, 4, 5, 6]
runs = list(range(10))

for mf_set in mf_sets_to_use:
    print("-------------")
    print(f"Running with mf_set={mf_set}")
    for n_similar_datasets in similar_datasets:
        for run in runs:
            for f in files:
                df = pd.read_csv(real_world_path + f)
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                print("---------------------------------")
                print(f"Running on dataset: {f}")
                print(f"Shape is: {X.shape}")

                X = X.to_numpy()
                y = y.to_numpy()
                #if X.shape[0] != 20000:
                #    continue
                if X.shape[0] <= 20000:
                    mf_set_string = Helper.mf_set_to_string(mf_set)
                    result_path = Path(path_to_store_results) / "similar_datasets" / f"n_similar_{n_similar_datasets}" / f"run_{run}" / Path(mf_set_string)

                    if not result_path.exists():
                        result_path.mkdir(exist_ok=True, parents=True)
                        
                    # Run our approach
                    n_warmstarts = 50 # Number of warmstart configurations (has to be smaller than n_loops)

                    n_loops = 50 # Number of optimizer loops. This is n_loops = n_warmstarts + x
                    limit_cs = True # Reduces the search space to suitable algorithms, dependening on warmstart configurations
                    time_limit = 120 * 60 # Time limit of overall optimization --> Aborts earlier if n_loops not finished but time_limit reached
                    cvi = "predict" # We want to predict a cvi based on our meta-knowledge 
                    dataset_name = f

                    # Instantiate our approach
                    ML2DAC = ApplicationPhase(mkr_path=mkr_path, mf_set=mf_set)
                    try:
                        optimizer_instance, additional_info = ML2DAC.optimize_with_meta_learning(X, n_warmstarts=n_warmstarts,
                                                                               n_optimizer_loops=n_loops, 
                                                                               limit_cs=limit_cs,
                                                                               cvi=cvi,
                                                                               time_limit=time_limit,
                                                                               dataset_name=dataset_name,
                                                                            n_similar_datasets=n_similar_datasets)
                        print(additional_info)

                        optimizer_result_df = process_result_to_dataframe(optimizer_instance, additional_info)

                        # Cleanup optimizer directory
                        #clean_up_optimizer_directory(optimizer_instance)

                    except FirstRunCrashedException as e:
                        print(e)
                        print("Generating empty file and skipping")
                        optimizer_result_df = pd.DataFrame()
                        optimizer_result_df[e] = True



                    optimizer_result_df.to_csv(result_path / dataset_name, index=False)
           


# # AutoML4Clust

# In[16]:


path_to_store_results = Path("real_world_results/Baselines/AML4C")

for f in files:
    df = pd.read_csv(real_world_path + f)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print("---------------------------------")
    print(f"Running on dataset: {f}")
    print(f"Shape is: {X.shape}")

    X = X.to_numpy()
    y = y.to_numpy()
    
    if X.shape[0] > 10000000000:
        print(f"Continue for {f}")
        continue
        
    # AML4C
    n_warmstarts = 0 # Number of warmstart configurations (has to be smaller than n_loops)
    n_loops = 50 # Number of optimizer loops. This is n_loops = n_warmstarts + x
    limit_cs = False # Reduces the search space to suitable algorithms, dependening on warmstart configurations
    time_limit = 120 * 60 # Time limit of overall optimization --> Aborts earlier if n_loops not finished but time_limit reached
    #cvi = "predict" # We want to predict a cvi based on our meta-knowledge 
    dataset_name = f

    #for cvi in CVIHandler.CVICollection.internal_cvis:
    for cvi in CVIHandler.CVICollection.internal_cvis:
        ML2DAC = ApplicationPhase(mkr_path=mkr_path, mf_set=mf_set)
        print("Starting Optimization")
        optimizer_instance, additional_info = ML2DAC.optimize_with_meta_learning(X,
                                                                               n_warmstarts=n_warmstarts,
                                                                   n_optimizer_loops=n_loops, 
                                                                   limit_cs=limit_cs,
                                                                   cvi=cvi,
                                                                   time_limit=time_limit,
                                                                   dataset_name=dataset_name)
        optimizer_result_df = process_result_to_dataframe(optimizer_instance, additional_info)

        print("Finished Optimization - Cleaning up directory")

        # Cleanup optimizer directory
        clean_up_optimizer_directory(optimizer_instance)

        mf_set_string = Helper.mf_set_to_string(mf_set)
        result_path = Path(path_to_store_results) / Path(cvi.get_abbrev())

        print(f"Storing result to {result_path}")
        if not result_path.exists():
            result_path.mkdir(exist_ok=True, parents=True)

        optimizer_result_df.to_csv(result_path / dataset_name, index=False)

        print("---------------------------------")


# # AutoClust (MLP-Model)
# 
# We require the mlp model in "/volume/related_work".

# In[35]:


import ast
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from ClusterValidityIndices.CVIHandler import CVICollection, MLPCVI
from ClusteringCS import ClusteringCS
from Experiments import DataGeneration
from MetaLearning import MetaFeatureExtractor
# define random seed
from MetaLearning.MetaFeatureExtractor import load_kdtree, query_kdtree
from Optimizer.OptimizerSMAC import SMACOptimizer

np.random.seed(1234)

related_work_path = Path("/volume/related_work")
n_loops = 10

def train_mlp_model(related_work_offline_result):
    # get the training data for mlp
    X = related_work_offline_result[[metric.get_abbrev() for metric in CVICollection.internal_cvis]].to_numpy()
    y = related_work_offline_result['ARI'].to_numpy()
    
    print(related_work_offline_result)
    print(X)
    # train the mlp
    mlp = MLPRegressor(hidden_layer_sizes=(60, 30, 10), activation='relu')
    mlp.fit(X, y)
    y_pred = mlp.predict(X)

    print(f"Score is: {r2_score(y, y_pred)}")
    return mlp

def store_mlp_model(mlp):
    with open(rw_mlp_filename, 'wb') as file:
        pickle.dump(mlp, file)

        
def run_online_on_dataset(X, y, name, mlp):
    ###################################################################
    # 1.) Algorithm Selection
    # find most similar dataset
    print(f"Using dataset to query: {name}")

    # 1.1) extract meta-features
    t0 = time.time()
    names, meta_features = MetaFeatureExtractor.extract_landmarking(X, mf_set="meanshift")
    mf_time = time.time() - t0

    # 1.2) load kdtree
    tree = load_kdtree(path=related_work_path, mf_set='meanshift')
    
    # 1.3) find nearest neighbors
    dists, inds = query_kdtree(meta_features, tree, k=len(d_names))
    print(f"most similar datasets are: {[d_names[ind] for ind in inds[0]]}")

    inds = inds[0]
    dists = dists[0]

    print(dists)

    # 1.4) assign distance column and filter such that the same dataset is not used. Note that for mf extraction,
    # the same dataset does not necessarily have distance=0
    dataset_name_to_distance = {d_name: dists[ind] for ind, d_name in enumerate(d_names)}
    rw_opt_result_for_dataset = related_work_offline_result[related_work_offline_result['dataset'] != name]
    print(rw_opt_result_for_dataset['dataset'].unique())
    rw_opt_result_for_dataset['distance'] = [dataset_name_to_distance[dataset_name] for dataset_name
                                             in rw_opt_result_for_dataset['dataset']]
    # assign algorithm column
    rw_opt_result_for_dataset["algorithm"] = rw_opt_result_for_dataset.apply(
        lambda x: ast.literal_eval(x["config"])["algorithm"],
        axis="columns")
    # sort first for distance and then for the best ARI score
    rw_opt_result_for_dataset = rw_opt_result_for_dataset.sort_values(['distance', 'ARI'], ascending=[True, False])

    # 1.5) get best algorithm
    best_algorithm = rw_opt_result_for_dataset['algorithm'].values[0]

    ###############################################################################
    # 2.) HPO with/for best algorithm
    # 2.1) build config space for algo
    best_algo_cs = ClusteringCS.build_paramter_space_per_algorithm()[best_algorithm]

    # 2.2) Use custom defined Metric for the mlp model --> Use this as metric for optimizer
    mlp_metric = MLPCVI(mlp_model=mlp)

    # 2.3) Optimize Hyperparameters with the mlp metric
    opt_instance = SMACOptimizer(dataset=X, true_labels=y,
                                 cvi=mlp_metric,
                                 n_loops=n_loops,
                                 cs=best_algo_cs)
    opt_instance.optimize()

    # 3.) Retrieving and storing result of optimization
    return opt_instance


# In[39]:


related_work_offline_result = pd.read_csv(related_work_path / 'related_work_offline_opt.csv', index_col=None)
print(related_work_offline_result.isna().sum())
related_work_offline_result = related_work_offline_result.dropna(how="any")
mlp_model = train_mlp_model(related_work_offline_result)

result_path = Path("real_world_results/Baselines/AutoClust")
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
    
    if X.shape[0] <= 10000000000:
        autoclust_result = run_online_on_dataset(X, y, f, mlp_model)
        additional_info = {"cvi": "MLP"}
        autoclust_df = process_result_to_dataframe(autoclust_result, additional_info)
        autoclust_df.to_csv(result_file, index=False)
        


# In[ ]:





# # AutoCluster

# In[24]:


import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ClusteringCS import ClusteringCS
from MetaLearning import MetaFeatureExtractor
from Experiments import DataGeneration

from MetaLearning.MetaFeatureExtractor import extract_all_datasets, load_kdtree, query_kdtree
from ClusterValidityIndices.CVIHandler import CVICollection
from Optimizer.OptimizerSMAC import SMACOptimizer
import os


# In[28]:


def process_result_to_dataframe_autocluster(optimizer_result, additional_info):
    selected_cvi = additional_info["cvi"]
    # The result of the application phase an optimizer instance that holds the history of executed
    # configurations with their runtime, cvi score, and so on.
    # We can also access the predicted clustering labels of each configuration to compute ARI.
    optimizer_result_df = optimizer_result.get_runhistory_df()
    for key, value in additional_info.items():
        if key == "algorithms":
            value = "+".join(value)
        optimizer_result_df[key] = value

    #optimizer_result_df = Helper.add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
    optimizer_result_df["iteration"] =  [i + 1 for i in range(len(optimizer_result_df))]
    optimizer_result_df["wallclock time"] =  optimizer_result_df["runtime"].cumsum()

    optimizer_result_df['CVI score'] = optimizer_result_df[selected_cvi]
    optimizer_result_df['Best CVI score'] = optimizer_result_df[selected_cvi].cummin()
    optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, y)
    optimizer_result_df['Best ARI'] = optimizer_result_df.apply(
        lambda row: 
        # Get ARI value of same rows with best CVI score, but the first one --> This is the one with the actual best CVI score
        optimizer_result_df[(optimizer_result_df["Best CVI score"]==row['Best CVI score'])]["ARI"].values[0],
        axis=1)

    print(optimizer_result_df)

    # We do not need the labels in the CSV file
    optimizer_result_df = optimizer_result_df.drop(selected_cvi, axis=1)
    print(optimizer_result_df)
    return optimizer_result_df


# In[21]:


def run_autoCluster_for_dataset(X, y, d_name):
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
    rw_opt_result_for_dataset = rw_opt_result_for_dataset.sort_values(['distance', cvi.get_abbrev()], ascending=[True, False])

    best_algorithm = rw_opt_result_for_dataset['algorithm'].values[0]
    
    ###############################################################################
    # 2.) HPO with/for best algorithm
    # 2.1) build config space for algo
    best_algo_cs = ClusteringCS.build_paramter_space_per_algorithm()[best_algorithm]
    
    # 2.3) Optimize Hyperparameters with the CVI --> Actually, they would perform a grid search, but thats too time-consuming
    opt_instance = SMACOptimizer(dataset=X, true_labels=y,
                                 cvi=cvi,
                                 n_loops=50, cs=best_algo_cs)
    opt_instance.optimize()
    
    return opt_instance


# In[33]:


related_work_path = Path("/volume/related_work")

path_to_store_results = Path("real_world_results/Baselines/AutoCluster")
related_work_offline_result = pd.read_csv(related_work_path / 'related_work_offline_opt.csv', index_col=None)
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
    
    
    if X.shape[0] <= 10000000000:
        
        # We get an ensemble with n X #cvis, i.e., each CVI predicts a clustering label for each data instance
        ensemble = np.zeros((X.shape[0], len(cvis_to_use)))
        for cvi in cvis_to_use:
            result_path = path_to_store_results / Path(cvi.get_abbrev())
            if not result_path.exists():
                result_path.mkdir(exist_ok=True, parents=True)
            result_file = result_path / Path(f)
            autocluster_result = run_autoCluster_for_dataset(X, y, f)
            additional_info = {"cvi": cvi.get_abbrev()}
            
            #autoclust_df.to_csv(result_file, index=False)
            autocluster_cvi_df = process_result_to_dataframe_autocluster(autocluster_result, additional_info)
            autocluster_cvi_df.to_csv(result_file, index=False)
            clean_up_optimizer_directory(autocluster_result)

