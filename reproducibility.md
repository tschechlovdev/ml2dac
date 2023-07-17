# Reproducibility for ML2DAC: Meta-learning to Democratize AutoML for Clustering Analyses

## Hardware Requirements 

* 16GB RAM
* 8 CPU cores 
* 10 GB Disk space

## Software Requirements

Only a running Docker installation is required. The scripts have been tested with the Versions: 

* macOS: 20.10.11
* Ubuntu: XXXX
* Windows: XXXX

## Prerequisites 

A docker installation is required, to install docker please follow these steps:

* [Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
* [macOS](https://docs.docker.com/desktop/install/mac-install/)
* [Windows](https://docs.docker.com/desktop/install/windows-install/)

## Reproduce the Experiments 

### Build the Container


First the docker images needs to be build. To do this, navigate into the root directory of the project and run:

```
docker build -t ml2dac .
```

### Run the Experiments

Afterwards the container can be run, please note, the default execution to fully reproduce the results may take more than 24 hours. Therefore, we provide some parameters to reduce the runtime or just change the experiment setting. This will have an effect on the results. 

Whenever the container is run, a result path on the host machine has to be defined. This is done by passing the parameter `local/dir/output:/app/evaluation_results/plots_notebooks/output`. The first part (everything till the ':') needs to be altered to a location on your machine where you have writing permissions. 

A default execution would look like the following: 

```
docker run -v local/dir/output:/app/evaluation_results/plots_notebooks/output  ml2dac
```

In order to reduce runtime or only execute certain experiments, the following parameters are available: 

**Only Figures**

This parameter sets of only the figures are generated, if set to `true` the other parameters become irrelevant. The default is `false`.

* `-e ONLY_FIGURES=true`: generates only the figures
* `-e ONLY_FIGURES=false`: experiments 

**Experiment**

This parameter decides which experiments are executed. If all experiments should be executed, this parameter does not need to be defined.  The available values are RealWorld, Evaluation, VaryingTrainingData and Ablation. This can be passed as string but need to be comma separated: 

* `-e EXPERIMENTS=RealWorld,Evaluation,VaryingTrainingData,Ablation`: would execute all experiments 
* `-e EXPERIMENTS=RealWorld,Evaluation` only the RealWorld and Evaluation experiment 
* `-e EXPERIMENTS=RealWorld` just the RealWorld experiment 

**Additional Parameters**

* `-e WARMSTARTS=25`: Integer value, which defines the number of warmstarts 
* `-e LOOPS=100`: Integer value, which defines the number of loops 
* `-e RUNS=10`: Integer value, which defines the number of runs 


A full example which executes only the experiments RealWorld and Evaluation with the parameters *warmstarts=1*,*runs=1* and *loops=1* would look like: 

```
docker run -e ONLY_FIGURES=false -e EXPERIMENTS=RealWorld,Evaluation -e WARMSTARTS=1 -e LOOPS=1 -e RUNS=1 -v local/dir/output:/app/evaluation_results/plots_notebooks/output  ml2dac
```