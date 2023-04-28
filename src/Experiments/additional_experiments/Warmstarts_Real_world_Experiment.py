#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

warnings.filterwarnings(category=RuntimeWarning, action="ignore")
warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")
import numpy as np
np.random.seed(0)
# Specify where to find our MKR
mkr_path = Path("../MetaKnowledgeRepository/") 

mf_set = MetaFeatureExtractor.meta_feature_sets[2]

real_world_path = "real_world_data/"
path_to_store_results = Path("real_world_results/ML2DAC/warmstarts")
files = [f for f in os.listdir(real_world_path) if os.path.isfile(real_world_path+ f)]
for f in files:
    print(f)
    df = pd.read_csv(real_world_path + f)
    print(df.shape)
    print("--------")
    


# In[2]:


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVIHandler.CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


# In[3]:


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


# In[4]:


def clean_up_optimizer_directory(optimizer_instance):
    if os.path.exists(optimizer_instance.output_dir) and os.path.isdir(optimizer_instance.output_dir):
                shutil.rmtree(optimizer_instance.output_dir)


# # Our Approach (Warmstarts)

# In[8]:


from smac.tae import FirstRunCrashedException
mf_sets_to_use = [#MetaFeatureExtractor.meta_feature_sets[2], # "statistical"
                  MetaFeatureExtractor.meta_feature_sets[4], # ["statistical", "info-theory", "general"]
                  MetaFeatureExtractor.meta_feature_sets[5], # ["statistical", "general"]
                  MetaFeatureExtractor.meta_feature_sets[8] # "autocluster"
                 ]

print(mf_sets_to_use)
for mf_set in mf_sets_to_use:
    print("-------------")
    print(f"Running with mf_set={mf_set}")
    
    for f in files:
        df = pd.read_csv(real_world_path + f)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        print("---------------------------------")
        print(f"Running on dataset: {f}")
        print(f"Shape is: {X.shape}")

        X = X.to_numpy()
        y = y.to_numpy()
        if X.shape[0] <= 20001:
            # Run our approach
            n_warmstarts = 50 # Number of warmstart configurations (has to be smaller than n_loops)
            n_loops = n_warmstarts # Number of optimizer loops. This is n_loops = n_warmstarts + x
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
                                                                       dataset_name=dataset_name)
                print(additional_info)
            
                optimizer_result_df = process_result_to_dataframe(optimizer_instance, additional_info)
            
                # Cleanup optimizer directory
                clean_up_optimizer_directory(optimizer_instance)
                
            except FirstRunCrashedException as e:
                print(e)
                print("Generating empty file and skipping")
                optimizer_result_df = pd.DataFrame()
                optimizer_result_df[e] = True
                
            mf_set_string = Helper.mf_set_to_string(mf_set)
            result_path = Path(path_to_store_results) / Path(mf_set_string)
            
            if not result_path.exists():
                result_path.mkdir(exist_ok=True, parents=True)
            
            optimizer_result_df.to_csv(result_path / dataset_name, index=False)



