# ML2DAC: Meta-learning to Democratize AutoML for Clustering Analyses

This repository contains the prototypical implementation of our ML2DAC approach. To address the combined algorithm
selection and hyperparameter optimization, so-called CASH problem, analysts typically execute various clustering
configurations, i.e., clustering algorithms with their hyperparameter settings. Thereby, especially novice analysts face
several challenges. Hence, we use meta-learning to support especially novice analysts with these challenges, so that
they obtain valuable clustering results in a short time frame. We use meta-learning to (a) select a cluster validity
index, (b)  select configurations in form of warmstart configurations efficiently, and (c) select suitable clustering
algorithms.

## Reproducibility

For reproducing our experiments, please switch to the branch "reproducibility" and see the ``reproducibility.md``
for more detailed instructions.

### Overview

Our repository contains several modules. In the following, we briefly describe each of the modules in the "src/"
Directory:

- "ClusteringCS/": Defines the configuration space. Here, we rely on nine different sklearn clustering algorithms and
  their hyperparameters. See `ALGORITHMS_MAP` in `ClusteringCS` for more details regarding each algorithm and its
  hyperparameter space.
- "ClusterValidityIndices/": Contains the seven Cluster Validity Indices that we use in our evaluation. It is easy to
  add more indices in the `CVIHandler.py` file.
- "Examples/": Contains an example python Notebook that shows how to use our approach.
- "Experiments/": Contains the code for running our experiments from our paper. It also contains the code for
  synthetically generating the data that we use. The folder is further split into synthetic and real-world experiments.
  The implementations for the baselines are contained in the respective experiments folder.
- "MetaLearning": Contains the code for our learning phase (`LearningPhase.py`) and our application
  phase (`ApplicationPhase.py`). Further, it contains a script ``MetaFeatureExtractor.py`` that implements some of the
  metafeatures and uses the pymfe library to extract various sets of metafeatures.
- "MetaKnowledgeRepository": Here we provide the knowledge of our learning phase that we built and use in our
  experiments. To create the MetaKnowledgeRepository (MKR), the ``LearningPhase.py`` script has to be executed. MKR
  contains three parts: the (1) meta-features, (2) evaluated configurations, and (3) the classification mode (CM) to
  predict a CVI. The dataset repository that we use to create the MKR can be found in the script ``DataGeneration.py``.
  For more information of our learning phase and how to build the MKR, we refer to our paper.
- "Optimizer/": We implement different optimizers for clustering in this module using the SMAC library. For our
  evaluation, we rely on Bayesian optimization as it is a model-based optimizer and thus benefits from the warmstarting
  configurations.
- "Utils/": Contains several utility functions.

Furthermore, we provide the results of our evaluation (Section 7) as CSV files in the folder 
"ml2dac/evaluation_results".

### Installation

To install the ml2dac API, you require Python 3.9 and a Linux environment with Ubuntu >= 16.04. You can find and install
the required libraries from the `requirements.txt` file. To install you have to follow these steps:

- Clone this repo in a directory of your choice
- "sudo apt-get install build-essential swig"
- "sudo apt-get install python3.9-dev"
- Go into the ml2dac directory
- Add the project path to your system python path. On ubuntu you can achieve this by "export PYTHONPATH=$PYTHONPATH:
  /path/to/ml2dac/src"
- (Optional) If you want to use it as a python library within pip:
    - Run "python3 setup.py bdist_wheel"
    - There should be a "dist" directory in the ml2dac directory --> navigate into it
    - Run "pip3 install lib.wheel" where lib is something with ml2dac

Now you should be able to use the library. You can test it by checking out the examples.
