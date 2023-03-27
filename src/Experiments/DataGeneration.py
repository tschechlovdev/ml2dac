import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.utils import check_random_state
import pandas as pd

random_state = 1234
np.random.seed(random_state)

DATASET_TYPES = ['gaussian','circles', 'moons', 'varied']


def generate_datasets(dataset_types=DATASET_TYPES):
    different_shape_sets = {}
    # Definition of dataset characteristics --> Define number of instances, attributes, #clusters
    characteristics = {'k': [10, 30, 50],
                       'n': [
                             1000, 5000, 10000
                             ],
                       'd': [10, 30, 50],
                       'noise': [
                           0.0, 0.01,
                           0.05, 0.1],
                       'type': dataset_types}

    for n in characteristics['n']:
        for data_type in characteristics['type']:
            generator = check_random_state(random_state)

            if data_type == 'gaussian':
                noise = 0
                # gaussian and varied also have "k" value
                for k in characteristics['k']:
                    for d in characteristics['d']:
                        data = make_blobs(n_samples=n, n_features=d, centers=k, random_state=random_state)
                        if noise > 0:
                            # add sklearn methodology of adding noise --> Adds to EACH point a standard deviation
                            # given by noise.
                            data[0] += generator.normal(scale=noise, size=data[0].shape)
                        different_shape_sets[f"type={data_type}-k={k}-n={n}-d={d}-noise={noise}"] = data
            elif data_type == 'varied':
                noise = 0
                for k in characteristics['k']:
                    for d in characteristics['d']:
                        data = make_blobs(n_samples=n,
                                          n_features=d,
                                          centers=k,
                                          # varying cluster std for each cluster
                                          # Note that this is affected by "k"
                                          cluster_std=[0.5 + i / k for i in list(range(1, k + 1))],
                                          random_state=random_state)
                        if noise > 0:
                            # add sklearn methodology of adding noise --> Adds to EACH point a standard deviation
                            # given by noise.
                            data[0] += generator.normal(scale=noise, size=data[0].shape)

                        different_shape_sets[f"type={data_type}-k={k}-n={n}-d={d}-noise={noise}"] = data

            elif data_type == 'circles' or data_type == 'moons':
                k = 2
                d = 2
                for noise in characteristics['noise']:
                    if data_type == 'circles':
                        data = make_circles(n_samples=n, factor=0.5, noise=noise, random_state=random_state)
                    elif data_type == 'moons':
                        data = make_moons(n_samples=n, noise=noise, random_state=random_state)

                    different_shape_sets[f"type={data_type}-k={k}-n={n}-d={d}-noise={noise}"] = data
    return different_shape_sets


from sklearn.model_selection import train_test_split

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
    noise = splits [4].split("=")[1]

    characteristic_dict["dataset"] = data_name
    characteristic_dict["type"] = type
    characteristic_dict["k"] = k
    characteristic_dict["n"] = n
    characteristic_dict["f"] = f
    characteristic_dict["noise"] = noise

    df = df.append(characteristic_dict, ignore_index=True)


df_train, df_test = train_test_split(df, stratify=df[["type", "k", "noise"]], train_size=0.8)
print(df_train)
print(df_test)

print(df_train[df_train["type"]!= "gaussian"])

print(len(df_train[df_train["type"]!= "gaussian"]))