import json
import logging
import uuid
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Dict, Union, Type

import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from sklearn.preprocessing import StandardScaler
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario

from ClusterValidityIndices import CVIHandler
from ClusterValidityIndices.CVIHandler import CVICollection
from ClusteringCS.ClusteringCS import build_kmeans_space
from Optimizer.smac_function_ import smac_function


def preprocess_data(dataset):
    """

    :param dataset:
    :return:
    """
    return StandardScaler().fit_transform(dataset)


class OptimizerHistoryEntry:
    """
    Holds all the information that we want to track for each execution of a configuration during the optimization procedure.
    """

    def __init__(self, runtime, score, labels, config, budget=None, score_name='ARI', additional_metrics=None):
        # Important to note that runtime is the overall wallclock time
        self.runtime = runtime
        self.labels = labels
        self.score = score
        self.configuration = config
        self.score_name = score_name

        if additional_metrics:
            self.additional_metrics = additional_metrics
        else:
            self.additional_metrics = {}

    def to_dict(self, skip_labels=False):
        if skip_labels:
            return dict({'runtime': self.runtime, self.score_name: self.score,
                         'config': self.configuration}, **self.additional_metrics)
        else:
            return dict({'runtime': self.runtime, self.score_name: self.score,
                         'config': self.configuration, 'labels': self.labels}, **self.additional_metrics)

    def __str__(self):
        return f"OptimizerHistoryEntry(runtime={self.runtime}, score={self.score}, labels={len(self.labels)}, " \
               f"config={self.configuration})"

    def __repr__(self):
        return f'OptimizerHistoryEntry[runtime={self.runtime}, score={self.score}, labels={len(self.labels)},' \
               f'config={self.configuration}]'


class AbstractOptimizer(ABC):
    """
       Abstract Wrapper class for all implemented optimization methods.
       Basic purpose is to have convenient way of using the optimizers.
       Therefore, after initializing the optimizer, just by running the
       optimize function the best result found by the optimizer will be obtained.
    """
    n_loops = 50

    def __init__(self, dataset, cvi: Union[Type[CVIHandler.CVI], CVIHandler.MLPCVI] = CVICollection.CALINSKI_HARABASZ,
                 cs: Type[ConfigurationSpace] = None, n_loops=None, output_dir=None,
                 cut_off_time_seconds=5 * 60, wallclock_limit=120 * 60, true_labels=None,
                 random_sate=1234):
        """
        :param dataset: np.array of the dataset (without the labels)
        :param cvi: A metric from the MetricCollection. Default is CALINSKI_HARABASZ
        :param cs: ConfigurationSpace object that is used by the optimizer. If not passed, then default is used.
        You can also pass a string, i.e., "partitional" stands for three partitional clustering algorithms (GMM, kMeans, MiniBatchKMeans).
        The value "kmeans" stands for only the kmeans algorithm. The default for the k_range is (2, 200).
        You can also use the file ClusteringCS.py where the strings for the Configspace possibilities are contained.
        :param n_loops: Number of loops that the optimizer performs, i.e., the number of configurations to execute.
        :param output_dir: Output directory where smac stores the runhistory.
        """

        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = f"./smac/{self.get_abbrev()}/"

        if not cvi:
            self.cvi = CVICollection.CALINSKI_HARABASZ
        else:
            self.cvi = cvi

        self.dataset = preprocess_data(dataset)

        if not n_loops:
            # set default budget
            self.n_loops = 60
        else:
            self.n_loops = n_loops

        if not cs:
            # n_samples = self.dataset.shape[0]
            # n_features = self.dataset.shape[1]
            # build default config space
            self.cs = build_kmeans_space()
        elif isinstance(cs, ConfigurationSpace):
            self.cs = cs

        self.true_labels = true_labels
        self.optimizer_result = None
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.trajectory_dict_list = []

        # default cutoff time is 10 minutes. The cutoff time has to be given in seconds!
        self.cutoff_time = cut_off_time_seconds
        # maximum runtime for the whole optimization procedure in seconds. Default is 60 minutes (60 * 60 seconds)
        self.wallclock_limit = wallclock_limit
        self.random_state = random_sate

    def optimize(self):
        return self.optimizer_result

    @staticmethod
    @abstractmethod
    def get_name():
        pass

    @classmethod
    def get_abbrev(cls):
        return OPT_ABBREVS[cls.get_name()]

    def get_config_space(self):
        return self.cs


class SMACOptimizer(AbstractOptimizer):
    """
        State-of-the-Art Bayesian Optimizer.
         Can be configured to use Random Forests (TPE) or Gaussian Processes.
        However, Gaussian Processes are much slower and only work for low-dimensional parameter spaces.
        Due to this, the default implementation uses Random Forests.
    """

    def __init__(self, dataset, wallclock_limit=120 * 60, cvi=None, n_loops=None, smac=SMAC4HPO, cs=None,
                 ouput_dir=None, true_labels=None, random_state=1234):
        super().__init__(dataset=dataset, cvi=cvi, cs=cs, n_loops=n_loops, output_dir=ouput_dir,
                         true_labels=true_labels, wallclock_limit=wallclock_limit, random_sate=random_state)

        self.smac_algo = smac
        self.smac = None

    def optimize(self, initial_configs=None) -> Union[SMAC4HPO, SMAC4AC]:
        """
        Runs the optimization procedure. Requires that the optimizer is instantiated with a dataset and
        a suitable config space. The procedure returns the smac optimizer. However, you probably want to use the other
        methods the optimizer offers to get either the best configuration (get_incumbent)
        or to get the history (get_run_history).

        :param initial_configs: Initial configurations that the optimizer first selects in the first
        n_initial_configs loops. This can be useful for using meta-learning.
        :return:
        """
        tae_algorithm = partial(smac_function, optimizer_instance=self)

        # Scenario object
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": self.n_loops,  # max. number of function evaluations;
                             "cs": self.cs,  # configuration space
                             "deterministic": "true",
                             "output_dir": self.output_dir,
                             "cutoff": self.cutoff_time,
                             # max duration to run the optimization (in seconds)
                             "wallclock-limit": self.wallclock_limit,
                             "run_id": str(uuid.uuid4())
                             })

        n_workers = 1 if self.get_name() == SMACOptimizer.get_name() else -1
        self.smac = self.smac_algo(scenario=scenario,
                                   tae_runner=tae_algorithm,
                                   initial_configurations=None if not initial_configs else initial_configs,
                                   initial_design=None if initial_configs else SobolDesign,
                                   # intensifier=SuccessiveHalving,
                                   intensifier_kwargs=self.get_intensifier_kwargs(),
                                   n_jobs=n_workers,
                                   rng=self.random_state
                                   )
        print(self.cs)
        self.smac.optimize()

        return self.smac

    def get_incumbent(self) -> Configuration:
        return self.smac.solver.incumbent

    def get_incumbent_stats(self):
        # Retrieve runhistory
        runhistory = self.get_runhistory_df()
        # Get best config
        best_config = self.get_incumbent()
        # Translate config to dictionary
        best_config_dic = best_config.get_dictionary()
        # Find best result in runhistory
        best_config_stats = runhistory[runhistory["config"] == best_config_dic].to_dict('r')
        best_config_stats = best_config_stats[0]
        best_config_stats["labels"] = np.array(best_config_stats["labels"])
        return best_config_stats

    def set_trajectory(self, trajectory_dict_list):
        self.trajectory_dict_list = trajectory_dict_list

    def get_trajectory(self) -> List[Dict]:
        """
        Returns a list of dictionary which kept track of the "trajectory", i.e., of the incumbents.
        So how long it took to get the incumbent etc. Yet this only tracks the history for the best configuration
        at each loop, not the whole history.
        :return: List of dictionaries
        """
        if not self.trajectory_dict_list:
            self.trajectory_dict_list = [x._asdict() for x in self.smac.get_trajectory()]
        return self.trajectory_dict_list

    def get_run_history(self) -> List[OptimizerHistoryEntry]:
        """
        Returns a list of OptimizerHistoryEntry object, which contain the results of each configuration that the
         optimizer executed.
        :return:
        """
        if not self.optimizer_result:
            self.parse_result()

        return self.optimizer_result

    def get_run_hitory_list_dicts(self, skip_labels=False) -> List[Dict]:
        """
        Uses get_run_history to return a list of dictionaries where each dictionary in the list has the same keys.
        The reason for this is to easy export this to a csv file with a dataframe for example.
        :return:
        """
        if not self.optimizer_result:
            self.parse_result()
        history = self.get_run_history()
        return [opt_history_entry.to_dict(skip_labels=skip_labels) for opt_history_entry in history]

    def get_runhistory_df(self):
        return pd.DataFrame(data=self.get_run_hitory_list_dicts())

    def parse_result(self):
        """
        Should only be called after the optimize method was called. This method parses the "runhistory.json" and stores
        the information in the optimizer_result. The result can be get by calling the get_runhistory() method.
        """
        out_dir = self.smac.output_dir
        history = []

        with open('{}/runhistory.json'.format(out_dir)) as json_file:
            json_data = json.load(json_file)
            data = json_data['data']
            configs = json_data['configs']

            for runs in data:
                # parse budget
                budget = runs[0][3]

                conf_id = runs[0][0]
                config = configs[str(conf_id)]

                # parse metric score and runtime of evaluating the algorithm+metric
                run = runs[1]
                score = run[0]
                optimizer_time = run[1]

                # parse addititonal info like labels
                add_info = run[5]
                if "labels" in add_info:
                    y_pred = add_info["labels"]
                else:
                    # can happen that labels are not there, because the execution was cutoff
                    # or due to an error during executing the clustering algorithm
                    y_pred = [-2 for _ in self.dataset]

                additional_metrics = {}
                for internal_cvi in CVICollection.internal_cvis:
                    if internal_cvi.get_abbrev() in add_info:
                        metric_score = add_info[internal_cvi.get_abbrev()]
                        additional_metrics[internal_cvi.get_abbrev()] = metric_score

                # create entry object and save to list
                entry = OptimizerHistoryEntry(labels=y_pred, runtime=optimizer_time, score=score,
                                              config=config, additional_metrics=additional_metrics,
                                              score_name=self.cvi.get_abbrev())
                history.append(entry)

        self.optimizer_result = history

    def get_intensifier_kwargs(self):
        return {}

    @staticmethod
    def get_name():
        return "SMAC"


OPT_ABBREVS = {SMACOptimizer.get_name(): "BO"}
