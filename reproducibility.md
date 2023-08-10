# Reproducibility for "ML2DAC: Meta-learning to Democratize AutoML for Clustering Analyses"

This file contains the instructions and requirements to reproduce the results from our paper
"ML2DAC: Meta-learning to Democratize AutoML for Clustering Analyses".

## Hardware Requirements 

* 32GB RAM
* 8 CPU cores 
* 20 GB Disk space

## Software Requirements

Only a running Docker installation and git with git-lfs is required. 
The scripts have been tested with the Versions: 

* macOS: 20.10.11
* Ubuntu: 22.04 and 23.0

We do not support Windows, because the library that we use for the Optimizer ([SMAC](https://automl.github.io/SMAC3/v2.0.1)), does not support Windows.


### Docker Installation
To install Docker please follow these steps:

* [Ubuntu](https://docs.docker.com/engine/install/ubuntu/) (Not the Desktop Version)
* [macOS](https://docs.docker.com/desktop/install/mac-install/)

### Git and Git LFS

Furthermore, we require Git and Git Large File Storage (LFS) to be installed on the machine. 
Typically, Git is already installed on most machines. You can check that by running
```
git --version
```
If it does not output a Git version, you must first [install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

For Git LFS, you can find the instructions also [online](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux).
An instruction for installing Git LFS from the command line can be found [here](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md).
For Linux (apt/deb based systems), you just have to run 
````
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
````

and then install it using

```` 
sudo apt-get install git-lfs
````

You can  verify the git-lfs installation by running ``git lfs -v``.

## Setting up the project

You can clone the repository with 

````
git clone https://github.com/tschechlovdev/ml2dac.git
````

Go into the directory of the project

````
cd ml2dac
````

and then switch to the reproducibility branch 

````
git checkout -b reproducibility origin/reproducibility
````

Now we can initialize Git LFS and then pull the large files from the repository using
````
git lfs init
git lfs pull
````

Now the project should be set up.
The next step is to build the Docker image that simplifies the execution of the experiments.

## Reproduce the Experiments 

To reproduce the experiments, the Docker image has to be built and subsequently, the Docker container can be executed.

### Build the Docker Image

First, you have to build the Docker image.
To do this, navigate into the root directory (``\path\to\ml2dac``) of the project and run:

```
docker build -t ml2dac .
```

The process of building the image will take around 5 minutes.

#### Troubles Building the Docker image

**Permission Errors:**
In case you receive a permission error, you have to add the current user to the Docker group:
```
sudo usermod -aG docker $USER
```

Instead of logging out/in again, you can then run:

````
newgrp docker
````

**Timeout Error:** If the docker build command fails due to some Timeout error, you may have to add ``--network=host`` to the build command. So, you have to run:
```
docker build -t ml2dac . --network=host
``` 

**No Space Left on Device:** If you do not have any more space left on the device due to Docker containers, you can do some cleanup by running 

WARNING: Running Docker system prune --all will remove all unused containers, networks, images (including dangling images), build caches, and optionally, volumes. This can free up significant disk space but also remove important data and configurations. Before running this command, ensure you have proper backups and understand the potential impact on your system. Use with caution

```
docker system prune --all
```

### Run the Experiments

If the image is successfully built, you can run the container. 
Note that the default execution to reproduce all the results from the paper takes approximately 2 days.
Therefore, we provide some parameters to reduce the runtime or just change the experiment setting.
This might have an effect on the results, but the trends should be the same as in the paper.
For instance, we only execute one run for each experiment per default.
We will execute the experiments for reproducing figures 4 -7 and tables 3 - 6 from the evaluation in our paper. 

Whenever the container is run, a result path on the host machine has to be defined. 
This is done by passing the parameter `local/dir/output:/app/evaluation_results/plots_notebooks/output`.
The first part (``local/dir/output``) needs to be altered to a location on your machine where you have writing permissions. 

A default execution that executes all our experiments would look like the following: 

```
docker run -v local/dir/output:/app/evaluation_results/output  -d ml2dac
```

To verify that the experiments are executed, you can run 
````
docker ps
````

This should list the ml2dac image with status ``RUNNING``.
The container will run in the background, even if you close the connection to a virtual machine.
This will also output the tables and figures from our paper as soon as the process is finished.
You can find them in the specified output folder ``local/dir/output``.

In the default setting, we execute only one run, which already takes approximately 2 days.
Therefore, the figures and tables will probably not be exactly as they are shown in the paper, i.e., some small deviations might occur.
Nevertheless, the trends and key messages should be the same as reported in our paper.

#### Additional Parameters

In order to reduce runtime or only execute certain experiments, the following parameters are available:

**Only Figures:**
This parameter sets if only the figures are generated. If set to `true`, the other parameters become irrelevant. The default is `false`.

* `-e ONLY_FIGURES=true`: generates only the figures
* `-e ONLY_FIGURES=false`: executes the experiments and then creates figures (DEFAULT)

**Experiment:**
This parameter decides which experiments are executed. If all experiments should be executed, this parameter does not need to be defined. The available values are RealWorld, Evaluation, VaryingTrainingData, and Ablation. This can be passed as a string but need to be comma separated: 

* `-e EXPERIMENTS=RealWorld,Evaluation,VaryingTrainingData,Ablation`: would execute all experiments (DEFAULT).
* `-e EXPERIMENTS=Evaluation` executes the comparison on the synthetic data (Figure 4 - 6). 
* `-e EXPERIMENTS=RealWorld` executes only the comparison on the Real-world data  (Section 7.4: Figure 7 and Table 5)
* `-e EXPERIMENTS=Evaluation, RealWorld` executes the experiments for the comparison on synthetic and real-world data,
which are the most important results of our paper. This creates the results for Figures 4 - 7 and Table 5. 
* 
**Runs:** The parameter `-e RUNS=1` defines the number of runs to execute for each experiment. 


A full example that executes only the experiments RealWorld and Evaluation would look like this: 

```
docker run -e ONLY_FIGURES=false -e EXPERIMENTS=RealWorld,Evaluation -e RUNS=1 -v local/dir/output:/app/evaluation_results/output  -d ml2dac
```
