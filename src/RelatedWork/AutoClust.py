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


def rw_run_offline_phase():
    # we build a config space with the hyperparameters for each algorithm separately
    cs_per_algo = ClusteringCS.build_paramter_space_per_algorithm()
    clustering_algos = ClusteringCS.algorithms
    related_work_offline_result = pd.DataFrame()
    for algo in clustering_algos:
        for X, y, dataset_name in zip(datasets, true_labels, d_names):
            min_max_scaler = preprocessing.MinMaxScaler()
            X = min_max_scaler.fit_transform(X)
            # Run Optimization Procedure
            opt_instance = SMACOptimizer(dataset=X, true_labels=y, cvi=CVICollection.ADJUSTED_RAND,
                                         n_loops=100, cs=cs_per_algo[algo])
            opt_instance.optimize()
            rw_opt_result_df = opt_instance.get_runhistory_df()
            rw_opt_result_df['dataset'] = dataset_name
            rw_opt_result_df['algorithm'] = algo
            related_work_offline_result = pd.concat([related_work_offline_result, rw_opt_result_df])
            print(related_work_offline_result.columns)
            print(related_work_offline_result)

    related_work_offline_result.reset_index(inplace=True)
    related_work_offline_result.to_csv(related_work_path / 'related_work_offline_opt.csv', index=False)


def train_mlp_model(related_work_offline_result):
    # get the training data for mlp
    X = related_work_offline_result[[metric.get_abbrev() for metric in CVICollection.internal_cvis]].to_numpy()
    y = related_work_offline_result['ARI'].to_numpy()

    # train the mlp
    mlp = MLPRegressor(hidden_layer_sizes=(60, 30, 10), activation='relu')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    mlp.fit(X, y)
    y_pred = mlp.predict(X_test)

    print(f"Score is: {r2_score(y_pred, y_test)}")
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
    algorithms = rw_opt_result_for_dataset['algorithm'].unique()
    if X.shape[0] <= 20000 and algorithms[0] == ClusteringCS.SPECTRAL_ALGORITHM:
        best_algorithm = algorithms[1]
    else:
        best_algorithm = algorithms[0]


    ###############################################################################
    # 2.) HPO with/for best algorithm
    # 2.1) build config space for algo
    best_algo_cs = ClusteringCS.build_paramter_space_per_algorithm()[best_algorithm]

    # 2.2) Use custom defined Metric for the mlp model --> Use this as metric for optimizer
    mlp_metric = MLPCVI(mlp_model=mlp)

    # 2.3) Optimize Hyperparameters with the mlp metric
    opt_instance = SMACOptimizer(dataset=X, true_labels=y,
                                 cvi=mlp_metric,
                                 n_loops=100, cs=best_algo_cs)
    opt_instance.optimize()

    # 3.) Retrieving and storing result of optimization
    related_work_online_result_df = opt_instance.get_runhistory_df()

    related_work_online_result_df['iteration'] = [i + 1 for i in range(len(related_work_online_result_df))]

    related_work_online_result_df['metric'] = mlp_metric.get_abbrev()
    related_work_online_result_df['metric score'] = related_work_online_result_df[mlp_metric.get_abbrev()].cummin()
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
        related_work_online_result_df[mlp_metric.get_abbrev()] == related_work_online_result_df['metric score']]

    related_work_online_result_df['ARI'] = [CVICollection.ADJUSTED_RAND.score_cvi(data=None, labels=labels,
                                                                                  true_labels=y) for
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

    print(related_work_online_result_df['ARI'])
    related_work_online_result_df['dataset'] = name
    related_work_online_result_df['mf time'] = mf_time

    print(related_work_online_result_df.head())
    print(related_work_online_result_df.columns)

    print(related_work_online_result_df.iloc[related_work_online_result_df["MLP"].idxmin()])
    return related_work_online_result_df


def rw_run_online_phase():
    # 0.) Load MLP Model
    with open(rw_mlp_filename, 'rb') as file:
        mlp = pickle.load(file)

    related_work_online_result_all_datasets = pd.DataFrame()
    # for each dataset:
    for i in range(len(datasets)):
        # First, use one specific dataset for online phase
        dataset = datasets[i]
        y_true = true_labels[i]
        d_name = d_names[i]

        related_work_online_result_df = run_online_on_dataset(dataset, y_true, d_name, mlp)
        related_work_online_result_all_datasets = pd.concat(
            [related_work_online_result_all_datasets, related_work_online_result_df])
        related_work_online_result_all_datasets.to_csv(related_work_path / 'autoclust.csv')


different_shape_sets = DataGeneration.generate_datasets()

d_names = list(different_shape_sets.keys())
datasets = [X for X, y in different_shape_sets.values()]
true_labels = [y for X, y in different_shape_sets.values()]
from sklearn.datasets import load_iris, load_digits

X, y = load_iris(return_X_y=True)
X, y = load_digits(return_X_y=True)

df = pd.read_csv("/home/tschecds/automlclustering/src/Experiments/real_world_data/letter.csv")
X = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

print(df)
print(d_names)
print(len(d_names))
print(len(datasets))
rw_mlp_filename = related_work_path / "rw_mlp.pkl"

# actually we would extract meta-features --> should already be done by us
# extract_all_datasets(datasets=datasets,path=related_work_path, extract='meanshift_mf', d_names=d_names, save_metafeatures=True)

all_datasets_result_df = pd.DataFrame()
################################################################################################################
#########################Run Optimizer Offline Phase##########################################################

# rw_run_offline_phase()

# load offline optimizer result
related_work_offline_result = pd.read_csv(related_work_path / 'related_work_offline_opt.csv', index_col=None)
related_work_offline_result.dropna(inplace=True)
print(related_work_offline_result.isna().sum())

# train model
mlp_model = train_mlp_model(related_work_offline_result)

# store model
store_mlp_model(mlp=mlp_model)
################## Online Phase #####################################

#####################################################################

# rw_run_online_phase()
run_online_on_dataset(X, y, "letter", mlp_model)
# related_work_online_result_all_datasets = pd.read_csv(related_work_path / 'related_work_online_result.csv',
#                                                      index_col=None)
