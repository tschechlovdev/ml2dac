import ast
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors._kd_tree import KDTree

from ClusterValidityIndices import CVIHandler
from ClusterValidityIndices.CVIHandler import CVICollection
from Experiments.DataGeneration import generate_datasets
from MetaLearning import MetaFeatureExtractor, LearningPhase
from MetaLearning.ApplicationPhase import ApplicationPhase
from MetaLearning.LearningPhase import mkr_path, optimal_cvi_file_name
from MetaLearning.MetaFeatureExtractor import extract_meta_features
from Optimizer.OptimizerSMAC import SMACOptimizer

warnings.filterwarnings("ignore")
mkr_path = LearningPhase.mkr_path


def _remove_duplicates_from_ARI_s(ARI_s):
    ARI_s["algorithm"] = ApplicationPhase._assign_algorithm_column(ARI_s)
    ARI_s = ARI_s.drop_duplicates(subset="config", keep='first')
    ARI_s = ARI_s.drop_duplicates(subset=["algorithm", "ARI"], keep='first')
    ARI_s = ARI_s.drop("algorithm", axis=1)
    return ARI_s


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVIHandler.CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


def process_result_to_dataframe(optimizer_result, additional_info, ground_truth_clustering):
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
    optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, ground_truth_clustering)
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


n_warmstarts = 25
n_loops = 100
limit_cs = True
result_path = Path("gen_results/evaluation_results/synthetic_data/vary_training_data")

mf_set = MetaFeatureExtractor.meta_feature_sets[5]  # Mf-Set: Stats+General

shape_sets = generate_datasets()

dataset_names = shape_sets.keys()

df = pd.DataFrame()
for data_name in dataset_names:
    characteristic_dict = {}
    splits = data_name.split("-")
    type = splits[0].split("=")[1]
    k = splits[1].split("=")[1]
    n = splits[2].split("=")[1]
    f = splits[3].split("=")[1]
    noise = splits[4].split("=")[1]

    characteristic_dict["dataset"] = data_name
    characteristic_dict["type"] = type
    characteristic_dict["k"] = k
    characteristic_dict["n"] = n
    characteristic_dict["f"] = f
    characteristic_dict["noise"] = noise

    df = df.append(characteristic_dict, ignore_index=True)

df_train, df_test = train_test_split(df, stratify=df[["type", "k", "noise"]], train_size=0.8, random_state=1234)
print(df_train)
print(df_test)

print(df_train[df_train["type"] != "gaussian"])

print(len(df_train[df_train["type"] != "gaussian"]))

evaluated_configs = pd.read_csv(mkr_path / "evaluated_configs.csv")

evaluated_configs = evaluated_configs.drop("Unnamed: 0", axis=1)
print(evaluated_configs)

test_data = df_test.iloc[0]
runs = 1

for run in range(runs):
    random_seed = (runs + 1) * 1234
    for i in range(len(df_test)):
        test_data = df_test.iloc[i]
        dataset_name = test_data["dataset"]
        additional_result_info = {"dataset": dataset_name}

        test_X, test_y = shape_sets[dataset_name]

        test_type = test_data["type"]

        # Set training set by leaving out the same family
        # selected_training_data_df = df_train[df_train["type"] != test_type]
        selected_training_data_df = df_train

        selected_training_data_df = selected_training_data_df.reset_index()
        selected_training_data_df = selected_training_data_df.drop("index", axis=1)

        # If not using the same family
        # training_datasets_df = df_train

        # Extract Meta-Features from test dataset
        mf_names, mfs = extract_meta_features(shape_sets[dataset_name][0], mf_set)
        mfs_test_dataset = mfs

        # Extract Meta-Features from training datasets
        mfs_all_training_datasets = []
        for X, name in zip([shape_sets[d_name][0] for d_name in selected_training_data_df["dataset"]],
                           selected_training_data_df["dataset"].values):
            print(f"extracting metafeatures {mf_set} from dataset {name}")
            _, scores = extract_meta_features(X, mf_set)
            mfs_all_training_datasets.append(scores)

        mfs_all_training_datasets = np.array(mfs_all_training_datasets)

        # Use these meta-features to define a distance
        tree = KDTree(mfs_all_training_datasets)
        dist, ind = tree.query(mfs_test_dataset.reshape(1, -1), k=len(mfs_all_training_datasets))
        training_data_indices = ind[0]
        dist = dist[0]

        selected_training_data_df["distance"] = 0
        selected_training_data_df["training_index"] = 0

        # Assign distance and training_index column to training_data
        for dist_index, i in enumerate(training_data_indices):
            selected_training_data_df.at[i, "distance"] = dist[dist_index]
            selected_training_data_df.at[i, "training_index"] = i

        n_trainig_data_values = list(range(10, len(training_data_indices), 5))
        if len(training_data_indices) not in n_trainig_data_values:
            n_trainig_data_values.append(len(training_data_indices))
        n_trainig_data_values = list(range(2, 65, 10))

        for n_training in n_trainig_data_values:
            print(f"Running with #training data = {n_training}")
            file_name = dataset_name + ".csv"
            path = result_path / f"run_{run}" / f"n_training_data_{n_training}"

            if not path.exists():
                path.mkdir(exist_ok=True, parents=True)

            if (path / file_name).is_file():
                print(f"Continue {n_training} for dataset {dataset_name}")
                print("File already exists")
                print("------------------------------")
                continue

            # Select training datasets to use
            # --> n_training most-dissimilar datasets, i.e., with highest distance!
            final_training_data = selected_training_data_df.sort_values("distance",
                                                                        ascending=False).head(n_training)
            print(final_training_data)

            names_final_training_datasets = final_training_data["dataset"].unique()

            # Get training data from MKR --> Only use the available training data
            EC_selected_training_data = evaluated_configs[
                evaluated_configs["dataset"].isin(names_final_training_datasets)]

            # Get class labels for RandomForest Model (Best CVI for each dataset)
            optimal_cvi_per_dataset = pd.read_csv(mkr_path / optimal_cvi_file_name)
            optimal_cvi_per_dataset = optimal_cvi_per_dataset.drop("Unnamed: 0", axis=1)

            optimal_cvi_per_dataset = optimal_cvi_per_dataset[optimal_cvi_per_dataset["dataset"].isin(
                names_final_training_datasets)
            ]

            optimal_cvi_per_dataset["dataset"] = pd.Categorical(optimal_cvi_per_dataset["dataset"],
                                                                categories=names_final_training_datasets,
                                                                ordered=True
                                                                )

            optimal_cvi_per_dataset = optimal_cvi_per_dataset.sort_values("dataset")

            for opt_cvi_d, final_train_d in zip(optimal_cvi_per_dataset["dataset"],
                                                names_final_training_datasets):
                print(opt_cvi_d)
                print(final_train_d)
                assert opt_cvi_d == final_train_d

            # Train classifier for training dataset
            mf_X = mfs_all_training_datasets[final_training_data["training_index"]]
            cvi_y = optimal_cvi_per_dataset["cvi"]
            cvi_classifier = RandomForestClassifier(random_state=random_seed)
            cvi_classifier.fit(mf_X, cvi_y)

            ### (a1) find similar dataset ###

            d_s = final_training_data.loc[
                final_training_data["distance"].idxmin(), "dataset"]
            additional_result_info["similar dataset"] = [d_s]

            ### (a2) select cluster validity index ###
            predicted_cvi = cvi_classifier.predict(mfs_test_dataset.reshape(1, -1))[0]
            additional_result_info["cvi"] = predicted_cvi
            predicted_cvi = CVICollection.get_cvi_by_abbrev(predicted_cvi)

            print(predicted_cvi)

            # Configs from most-similar data
            EC_s = EC_selected_training_data[EC_selected_training_data["dataset"] == d_s]

            ARI_s = EC_s[["config", "ARI"]]
            print(ARI_s)
            ARI_s = ApplicationPhase._remove_duplicates_from_ARI_s(ARI_s)

            ### (a3) select warmstart configurations ###
            warmstart_configs = ApplicationPhase.select_warmstart_configurations(ARI_s, n_warmstarts=n_warmstarts)

            ### (a4) definition of configurations space (dependent on warmstart configurations) ###
            cs, algorithms = ApplicationPhase(k_range=(2,100)).define_config_space(warmstart_configs, limit_cs=limit_cs)
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

            opt_instance = SMACOptimizer(dataset=test_X,
                                         true_labels=None,  # we do not have access to them in the application phase
                                         cvi=predicted_cvi,
                                         n_loops=n_loops,
                                         cs=cs,
                                         wallclock_limit=240 * 60,

                                         )

            opt_instance.optimize(initial_configs=warmstart_configs)

            optimizer_result_df = process_result_to_dataframe(opt_instance, additional_info=additional_result_info,
                                                              ground_truth_clustering=test_y)
            clean_up_optimizer_directory(opt_instance)

            optimizer_result_df.to_csv(path / file_name, index=False)

            print("----------------------------------")
            print("finished optimization")
            print(f"best obtained configuration is:")
            print(opt_instance.get_incumbent())
