import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

from ClusterValidityIndices.CVIHandler import CVICollection
from Experiments import DataGeneration
from MetaLearning import LearningPhase, MetaFeatureExtractor
from MetaLearning.ApplicationPhase import ApplicationPhase
from Utils import Helper

warnings.filterwarnings(category=RuntimeWarning, action="ignore")
warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")
warnings.filterwarnings(category=UserWarning, action="ignore")
np.random.seed(0)


def compute_ari_values(optimizer_result_df, ground_truth_labels):
    return optimizer_result_df["labels"].apply(
        lambda labels:
        CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels, true_labels=ground_truth_labels)
    )


if __name__ == '__main__':
    shape_sets = DataGeneration.generate_datasets()
    datasets = [dataset[0] for key, dataset in shape_sets.items()]
    dataset_names = list(shape_sets.keys())
    true_labels = [dataset[1] for key, dataset in shape_sets.items()]

    # Parameters for our experiments
    n_warmstarts = 25
    n_loops = 100
    limit_cs = True
    time_limit = 120 * 60
    cvi = "predict"
    mkr_path = Path("../MetaKnowledgeRepository/")

    path_to_store_results = Path("../evaluation_results")
    if not path_to_store_results.exists():
        path_to_store_results.mkdir(exist_ok=True, parents=True)

    for mf_set in MetaFeatureExtractor.meta_feature_sets:
        # DataFrame to store results. We store the results for each meta-feature in a separate CSV file.
        evaluation_results = pd.DataFrame()
        cvi_prediction_results = pd.DataFrame()

        # Create ML2DAC instance
        ML2DAC = ApplicationPhase(mkr_path=mkr_path, mf_set=mf_set)
        for dataset, ground_truth_labels, dataset_name in zip(datasets, true_labels, dataset_names):
            # Run the application phase of our approach for each "new" dataset.
            # Note that we have executed the learning phase for the new dataset as well,
            # however we skip the meta-knowledge for it. For this, we use the dataset_name.
            optimizer_instance, additional_result_info = ML2DAC.optimize_with_meta_learning(X=dataset,
                                                                                            dataset_name=dataset_name,
                                                                                            n_warmstarts=n_warmstarts,
                                                                                            n_optimizer_loops=n_loops,
                                                                                            cvi=cvi,
                                                                                            limit_cs=limit_cs,
                                                                                            time_limit=time_limit)
            selected_cvi = additional_result_info["CVI"]
            # The result of the application phase an optimizer instance that holds the history of executed
            # configurations with their runtime, cvi score, and so on.
            # We can also access the predicted clustering labels of each configuration to compute ARI.
            optimizer_result_df = optimizer_instance.get_runhistory_df()
            for key, value in additional_result_info.items():
                if key == "algorithms":
                    value = "+".join(value)
                optimizer_result_df[key] = value

            # Preprocess results, add some columns, prune to only have the best configurations over time
            optimizer_result_df = Helper.add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi)
            print(optimizer_result_df)

            # Compute ARI values for the evaluation as we have ground-truth clusterings
            optimizer_result_df['ARI'] = compute_ari_values(optimizer_result_df, ground_truth_labels)

            # As we have pruned for the best CVI results, we copy the iterations that did not improve with the best CVI
            # value. This way we can easily plot the data afterwards.
            optimizer_result_df = Helper.add_missing_iterations(optimizer_result_df, n_loops)
            optimizer_result_df = optimizer_result_df.drop("labels", axis=1)
            evaluation_results = pd.concat([evaluation_results, optimizer_result_df])

            # Store results for this meta-feature set
            mf_set_string = Helper.mf_set_to_string(mf_set)
            evaluation_results.to_csv(path_to_store_results / f"results_{mf_set_string}.csv", index=False)

            # Find out which CVI would be the optimal for this dataset
            optimal_cvis = pd.read_csv(mkr_path / LearningPhase.optimal_cvi_file_name)
            optimal_cvi = optimal_cvis[optimal_cvis["dataset"] == dataset_name]["cvi"].values[0]
            # Get correlations of all CVIs to ARI
            cvi_correlations = optimal_cvis["correlations"].values[0]
            # Store CVI ranking separately
            cvi_prediction_results.append({"dataset": dataset_name, "Predicted CVI": selected_cvi,
                                           "Optimal CVI": optimal_cvi, "Correlations": cvi_correlations},
                                          ignore_index=True)
            cvi_prediction_results.to_csv(path_to_store_results / f"cvi_ranking_{mf_set_string}.csv", index=False)

            # Cleanup optimizer directory
            if os.path.exists(optimizer_instance.output_dir) and os.path.isdir(optimizer_instance.output_dir):
                shutil.rmtree(optimizer_instance.output_dir)
