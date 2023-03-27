from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd


def print_timestamp(s):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y - %H:%M:%S")
    print("{}: {}".format(date_time, s))


def mf_set_to_string(mf_set):
    return '+'.join(mf_set) if isinstance(mf_set, list) else mf_set


def get_model_name(classifier_instance=RandomForestClassifier()):
    return str(type(classifier_instance)).split(".")[-1][:-2]


def hopkins(data_frame, sampling_size):
    """
    Code taken from pyclustertend: https://github.com/lachhebo/pyclustertend/blob/master/pyclustertend/hopkins.py
    Slightly adjusted
    Assess the clusterability of a dataset. A score between 0 and 1, a score around 0.5 express
    no clusterability and a score tending to 0 express a high cluster tendency.

    Parameters
    ----------
    data_frame : numpy array
        The input dataset
    sampling_size : int
        The sampling size which is used to evaluate the number of DataFrame.

    Returns
    ---------------------
    score : float
        The hopkins score of the dataset (between 0 and 1)

    Examples
    --------
    >>> from sklearn import datasets
    >>> X = datasets.load_iris().data
    >>> hopkins(X,150)
    0.16
    """

    if type(data_frame) == np.ndarray:
        data_frame = pd.DataFrame(data_frame)

    # Sample n observations from D : P

    if sampling_size > data_frame.shape[0]:
        raise Exception(
            'The number of sample of sample is bigger than the shape of D')

    data_frame_sample = data_frame.sample(n=sampling_size)

    # Get the distance to their neirest neighbors in D : X

    tree = BallTree(data_frame, leaf_size=2)
    dist, _ = tree.query(data_frame_sample, k=2)
    data_frame_sample_distances_to_nearest_neighbours = dist[:, 1]

    # Randomly simulate n points with the same variation as in D : Q.

    max_data_frame = data_frame.max()
    min_data_frame = data_frame.min()

    uniformly_selected_values_0 = np.random.uniform(min_data_frame[0], max_data_frame[0], sampling_size)
    uniformly_selected_values_1 = np.random.uniform(min_data_frame[1], max_data_frame[1], sampling_size)

    uniformly_selected_observations = np.column_stack((uniformly_selected_values_0, uniformly_selected_values_1))
    if len(max_data_frame) >= 2:
        for i in range(2, len(max_data_frame)):
            uniformly_selected_values_i = np.random.uniform(min_data_frame[i], max_data_frame[i], sampling_size)
            to_stack = (uniformly_selected_observations, uniformly_selected_values_i)
            uniformly_selected_observations = np.column_stack(to_stack)

    uniformly_selected_observations_df = pd.DataFrame(uniformly_selected_observations)

    # Get the distance to their neirest neighbors in D : Y

    tree = BallTree(data_frame, leaf_size=2)
    dist, _ = tree.query(uniformly_selected_observations_df, k=1)
    uniformly_df_distances_to_nearest_neighbours = dist

    # return the hopkins score

    x = sum(data_frame_sample_distances_to_nearest_neighbours)
    y = sum(uniformly_df_distances_to_nearest_neighbours)

    if x + y == 0:
        raise Exception('The denominator of the hopkins statistics is null')

    return x / (x + y)[0]


def add_missing_iterations(optimizer_result_df, n_loops):
    """
    Adds the iterations that are not present due to the fact that we only track the best configurations.
    This is required for nicer plots.
    :param optimizer_result_df:
    :return: optimizer_result_df that now contains iterations from 1 to max_iterations in the dataframe
    """
    iterations = optimizer_result_df["iteration"].values
    if len(iterations > 0):
        last_iteration = 0

        for i in range(1, n_loops + 1):
            if i in iterations:
                last_iteration = i
            else:
                it_filtered = optimizer_result_df[optimizer_result_df["iteration"] == last_iteration]
                it_filtered["iteration"] = i
                print(it_filtered)
                optimizer_result_df = pd.concat([optimizer_result_df, it_filtered])
    return optimizer_result_df


def add_iteration_metric_wallclock_time(optimizer_result_df, selected_cvi):
    # We have the cvi name as column. However, we want a coulmn with metric and then the name of that metric
    optimizer_result_df['iteration'] = [i + 1 for i in range(len(optimizer_result_df))]
    optimizer_result_df['CVI'] = selected_cvi
    optimizer_result_df['CVI score'] = optimizer_result_df[selected_cvi].cummin()
    optimizer_result_df['wallclock time'] = optimizer_result_df['runtime'].cumsum()
    # set max_iteration --> need this to check for timeouts
    max_iteration = optimizer_result_df['iteration'].max()
    max_wallclock_time = optimizer_result_df['wallclock time'].max()
    optimizer_result_df['max wallclock'] = max_wallclock_time
    optimizer_result_df['max iteration'] = max_iteration
    optimizer_result_df = optimizer_result_df[optimizer_result_df[selected_cvi]
                                              == optimizer_result_df['CVI score']]


    return optimizer_result_df
