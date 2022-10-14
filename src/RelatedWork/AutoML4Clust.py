from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ClusteringCS import ClusteringCS

from Experiments import DataGeneration
from MetaLearning.MetricLearning import dataset_types, n_loops, time_limit
from ClusterValidityIndices.CVIHandler import CVICollection
from Optimizer.OptimizerSMAC import SMACOptimizer
from RelatedWork.AutoClust import related_work_path


def run_automl_four_clust():
    online_opt_result_all_datasets = pd.DataFrame()
    # todo: dont reuse results
    if Path(related_work_path / f'automl4clust.csv').is_file() and False:
        online_opt_result_all_datasets = pd.read_csv(related_work_path / f'automl4clust.csv', index_col=None)
        if "Unnamed: 0" in online_opt_result_all_datasets.columns:
            online_opt_result_all_datasets.drop('Unnamed: 0', axis='columns')
        if "Unnamed: 0.1" in online_opt_result_all_datasets.columns:
            online_opt_result_all_datasets.drop('Unnamed: 0.1', axis='columns')

    # We run per default AutoML4Clust with all CVIs
    coldstart_metrics = CVICollection.internal_cvis

    for coldstart_metric in coldstart_metrics:
        shape_sets = DataGeneration.generate_datasets(dataset_types=dataset_types)

        datasets_to_use = [dataset[0] for key, dataset in shape_sets.items()]
        dataset_names_to_use = list(shape_sets.keys())
        true_labels_to_use = [dataset[1] for key, dataset in shape_sets.items()]

        print("Using datasets: ")

        for dataset, dataset_name, dataset_labels, in zip(datasets_to_use, dataset_names_to_use, true_labels_to_use):
            cs = ClusteringCS.build_all_algos_space()
            metric_abbrev = coldstart_metric.get_abbrev()
            if 'dataset' in online_opt_result_all_datasets.columns:
                if len(online_opt_result_all_datasets[
                           (online_opt_result_all_datasets['dataset'] == dataset_name)
                           & (online_opt_result_all_datasets['metric'] == metric_abbrev)]) >= 1:
                    print(f"Dataset {dataset_name} already processed with metric {metric_abbrev}")
                    continue
            else:
                print(f"Dataset {dataset_name} not! processed with metric {metric_abbrev}")

            dataset = StandardScaler().fit_transform(dataset)

            opt_instance = SMACOptimizer(dataset=dataset, true_labels=dataset_labels,
                                         cvi=coldstart_metric,
                                         n_loops=n_loops, cs=cs, wallclock_limit=time_limit)

            opt_instance.optimize()
            online_opt_result_df = opt_instance.get_runhistory_df()

            online_opt_result_df['iteration'] = [i + 1 for i in range(len(online_opt_result_df))]

            # We have the metric name as column. However, we want a coulmn with metric and then the name of that metric
            online_opt_result_df['metric'] = metric_abbrev
            online_opt_result_df['metric score'] = online_opt_result_df[metric_abbrev].cummin()
            online_opt_result_df['wallclock time'] = online_opt_result_df['runtime'].cumsum()

            # set max_iteration --> need this to check for timeouts
            max_iteration = online_opt_result_df['iteration'].max()
            max_wallclock_time = online_opt_result_df['wallclock time'].max()
            online_opt_result_df['max wallclock'] = max_wallclock_time
            online_opt_result_df['max iteration'] = max_iteration

            # we only want to look at, where the incumbent changes! so where we get better metric values. Hence, the result
            # will not contain all performed iterations!
            online_opt_result_df = online_opt_result_df[
                online_opt_result_df[metric_abbrev] == online_opt_result_df['metric score']]

            online_opt_result_df['ARI'] = [CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels,
                                                                                 true_labels=dataset_labels) for
                                           labels
                                           in online_opt_result_df['labels']]
            iterations = online_opt_result_df["iteration"].values
            if len(iterations > 0):
                print(iterations)
                last_iteration = 0
                for i in range(1, 101):
                    if i in iterations:
                        last_iteration = i
                    else:
                        it_filtered = online_opt_result_df[
                            online_opt_result_df["iteration"] == last_iteration]
                        it_filtered["iteration"] = i
                        print(it_filtered)
                        online_opt_result_df = pd.concat([online_opt_result_df, it_filtered])

            online_opt_result_df = online_opt_result_df.drop([metric_abbrev, 'labels', 'budget'],
                                                             axis=1)

            print(f"ARI scores are: {online_opt_result_df['ARI']}")
            online_opt_result_df['dataset'] = dataset_name

            print(online_opt_result_df.head())
            print(online_opt_result_df.columns)

            online_opt_result_all_datasets = pd.concat([online_opt_result_all_datasets, online_opt_result_df])
            online_opt_result_all_datasets.to_csv(related_work_path / f'automl4clust.csv', index=False)