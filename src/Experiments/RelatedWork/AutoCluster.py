import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ClusteringCS import ClusteringCS
from Experiments.RelatedWork.AutoClust import related_work_result_path
from MetaLearning import MetaFeatureExtractor
from Experiments import DataGeneration

from MetaLearning.MetaFeatureExtractor import extract_all_datasets, load_kdtree, query_kdtree
from ClusterValidityIndices.CVIHandler import CVICollection
from Optimizer.OptimizerSMAC import SMACOptimizer
import os

from Experiments.RelatedWork import AutoClust

# define random seed
np.random.seed(1234)
n_loops = 100

# We only use the three CVIs defined in the Paper of Liu et al.
# Otherwise, it would be infeasible to run an optimizer loop for seven CVIs
cvis = [CVICollection.CALINSKI_HARABASZ, CVICollection.SILHOUETTE, CVICollection.DAVIES_BOULDIN]

related_work_path = AutoClust.related_work_path
Path(related_work_path).mkdir(exist_ok=True, parents=True)


# files = [f for f in os.listdir('../MetaLearning') if os.path.isfile(f)]
# for file in files:
#     print(file)


def rw_run_offline_phase():
    extract_all_datasets(datasets=datasets, path=related_work_path, mf_Set="autocluster", d_names=d_names,
                         save_metafeatures=True)


def run_online_phase_for_cvi(cvi):
    related_work_online_result_all_datasets = pd.DataFrame()
    for i in range(len(datasets)):
        # First, use one specific dataset for online phase
        dataset = datasets[i]
        y_true = true_label_set[i]
        d_name = d_names[i]

        ###################################################################
        # 1.) Algorithm Selection
        # find most similar dataset
        print(f"Using dataset to query: {d_name}")

        # 1.1) extract meta-features
        t0 = time.time()
        names, meta_features = MetaFeatureExtractor.extract_autocluster_mfes(dataset)
        mf_time = time.time() - t0

        # 1.2) load kdtree
        tree = load_kdtree(path=related_work_path, mf_set='autocluster')

        # 1.3) find nearest neighbors
        dists, inds = query_kdtree(meta_features, tree, k=len(d_names))
        print(f"most similar datasets are: {[d_names[ind] for ind in inds[0]]}")

        inds = inds[0]
        dists = dists[0]

        print(dists)

        # 1.4) assign distance column and filter such that the same dataset is not used. Note that for mf extraction,
        # the same dataset does not necessarily have distance=0
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

        # 1.5) get best algorithm
        algorithms = rw_opt_result_for_dataset['algorithm'].unique()
        if dataset.shape[0] <= 20000 and algorithms[0] == ClusteringCS.SPECTRAL_ALGORITHM:
            best_algorithm = algorithms[1]
        else:
            best_algorithm = algorithms[0]

        print(best_algorithm)
        ###############################################################################
        # 2.) HPO with/for best algorithm
        # 2.1) build config space for algo
        best_algo_cs = ClusteringCS.build_paramter_space_per_algorithm()[best_algorithm]

        # 2.3) Optimize Hyperparameters with the mlp metric
        opt_instance = SMACOptimizer(dataset=dataset, true_labels=y_true,
                                     cvi=cvi,
                                     n_loops=n_loops, cs=best_algo_cs)
        opt_instance.optimize()

        # 3.) Retrieving and storing result of optimization
        related_work_online_result_df = opt_instance.get_runhistory_df()

        related_work_online_result_df['iteration'] = [i + 1 for i in range(len(related_work_online_result_df))]

        related_work_online_result_df['metric'] = cvi.get_abbrev()
        related_work_online_result_df['metric score'] = related_work_online_result_df[cvi.get_abbrev()].cummin()
        related_work_online_result_df['wallclock time'] = related_work_online_result_df['runtime'].cumsum()
        related_work_online_result_df['algorihtm'] = best_algorithm

        # set max_iteration --> need this to check for timeouts
        max_iteration = related_work_online_result_df['iteration'].max()
        related_work_online_result_df['max iteration'] = max_iteration

        max_wallclock_time = related_work_online_result_df['wallclock time'].max()
        related_work_online_result_df['max wallclock'] = max_wallclock_time
        # we only want to look at, where the incumbent changes! so where we get better metric values. Hence, the result
        # will not contain all performed iterations!
        related_work_online_result_df = related_work_online_result_df[
            related_work_online_result_df[cvi.get_abbrev()] == related_work_online_result_df['metric score']]

        related_work_online_result_df['ARI'] = [CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels,
                                                                                      true_labels=y_true) for
                                                labels
                                                in related_work_online_result_df['labels']]
        iterations = related_work_online_result_df["iteration"].values
        if len(iterations > 0):
            print(iterations)
            last_iteration = 0
            for i in range(1, 101):
                if i in iterations:
                    last_iteration = i
                else:
                    it_filtered = related_work_online_result_df[
                        related_work_online_result_df["iteration"] == last_iteration]
                    it_filtered["iteration"] = i
                    print(it_filtered)
                    related_work_online_result_df = pd.concat([related_work_online_result_df, it_filtered])

        related_work_online_result_df = related_work_online_result_df.drop(
            [cvi.get_abbrev(),
             # 'labels', For AutoCluster we need the labels for consensus (majority voting)
             ],
            axis=1)

        print(related_work_online_result_df['ARI'])
        related_work_online_result_df['dataset'] = d_name
        related_work_online_result_df['mf time'] = mf_time

        print(related_work_online_result_df.head())
        print(related_work_online_result_df.columns)

        related_work_online_result_all_datasets = pd.concat(
            [related_work_online_result_all_datasets, related_work_online_result_df])
        related_work_online_result_all_datasets.to_csv(related_work_result_path / "autocluster_online.csv")
    return related_work_online_result_all_datasets


def rw_run_online_phase():
    # for each dataset:
    results = pd.DataFrame()
    for cvi in cvis:
        online_result_cvi_all_datasets = run_online_phase_for_cvi(cvi)
        results = pd.concat([results, online_result_cvi_all_datasets])
        results.to_csv(related_work_result_path / "autocluster_online.csv")


def run_majority_voting():
    online_result = pd.read_csv(related_work_result_path / "autocluster_online.csv")
    online_result["type"] = online_result["dataset"].apply(lambda data: data.split("-")[0].split("=")[1])
    online_result["Method"] = online_result["metric"].apply(lambda x: "AutoCluster - {}".format(x))
    piv_table = online_result.pivot(index=["dataset", "iteration"], columns=['metric'],
                                    values=['labels', 'wallclock time', 'ARI'])
    online_result_copy = online_result.copy()
    runtime_overhead = 0

    for i in range(1, 101):
        for k in range(len(datasets)):
            # First, use one specific dataset for online phase
            # X = datasets[k]
            y_true = true_label_set[k]
            data = d_names[k]

            print(f"Processing line {i * d_names.index(data)} of {len(piv_table)}")
            new_data_row = {"dataset": data,
                            "wallclock time": piv_table["wallclock time"].loc[(data, i)].sum(),
                            "iteration": i, "metric": "MV"}

            true_labels = y_true
            sil_labels = ast.literal_eval(piv_table["labels"]["SIL"].loc[(data, i)])
            ch_labels = ast.literal_eval(piv_table["labels"]["CH"].loc[(data, i)])
            dbi_labels = ast.literal_eval(piv_table["labels"]["DBI"].loc[(data, i)])
            # print(ch_labels)

            start = time.time()
            labels = np.array([sil_labels, ch_labels, dbi_labels])
            from collections import Counter
            co_assoc_matrix = labels
            final_result = []

            for j in range(len(co_assoc_matrix[0])):
                cluster_labels = co_assoc_matrix[:, j]
                idx = np.argmax(list(Counter(cluster_labels).values()))
                final_result.append(cluster_labels[idx])
            # final_result = [ for i in range()],
            # np.array(final_result)

            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(final_result, true_labels)
            runtime = time.time() - start
            runtime_overhead += runtime
            new_data_row["ARI"] = ari * -1

            online_result_copy = online_result_copy.append(new_data_row, ignore_index=True)

    print(f"Overall Runtime overhead for majority voting: {runtime_overhead}")
    online_result_copy.to_csv(related_work_path / "autocluster_mv.csv", index=None)


def run_autocluster_experiments():
    related_work_offline_result = pd.read_csv(related_work_path / 'related_work_offline_opt.csv', index_col=None)
    different_shape_sets = DataGeneration.generate_datasets()

    d_names = list(different_shape_sets.keys())
    datasets = [X for X, y in different_shape_sets.values()]
    true_label_set = [y for X, y in different_shape_sets.values()]
    print(d_names)
    print(len(d_names))
    print(len(datasets))

    ################## Online Phase #####################################

    #####################################################################

    rw_run_online_phase()
    run_majority_voting()


related_work_offline_result = pd.read_csv(related_work_path / 'related_work_offline_opt.csv', index_col=None)
different_shape_sets = DataGeneration.generate_datasets()
d_names = list(different_shape_sets.keys())
datasets = [X for X, y in different_shape_sets.values()]
true_label_set = [y for X, y in different_shape_sets.values()]

if __name__ == '__main__':
    run_autocluster_experiments()
