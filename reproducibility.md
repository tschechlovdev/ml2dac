# Reproducibility for ML2DAC: Meta-learning to Democratize AutoML for Clustering Analyses

## Hardware Requirements 

* 32GB RAM
* 8 CPU cores 
* 20 GB Disk space

## Software Requirements

Only a running Docker installation is required. The scripts have been tested with the Versions: 

* macOS: 20.10.11
* Ubuntu: 23.0.5


## Prerequisites 

A docker installation is required, to install docker please follow these steps:

* [Ubuntu](https://docs.docker.com/engine/install/ubuntu/) Not the Desktop Version
* [macOS](https://docs.docker.com/desktop/install/mac-install/)
* [Windows](https://docs.docker.com/desktop/install/windows-install/)

However, note that the installation of Docker is more straightforwad on Ubuntu and macOS than on Windows.

Furthermore, we require git and git-lfs to be installed on the machine. 
Typically, git is already installed on most machines. You can check that by running
```
git --version
```
If it does not output a git version, then you first have to [install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

For git-lfs, you can find the instructions also [online](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux).
An instruction for installing git-lfs from the command line can be found [here](https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md).
For Linux (apt/deb based systems), you just have to run 
````
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
````

and then install it using

```` 
sudo apt-get install git-lfs
````

You can  verify the git-lfs installation by running ``git lfs -v`` .



## Setup the project

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

Now we can initialize git lfs and then pull the large files from the repository using
````
git lfs init
git lfs pull
````

Now the project should be setup.
The next step is to build the Docker image that simplifies the execution of the experiments.

## Reproduce the Experiments 

### Build the Container

First you have to build the docker imag.
To do this, navigate into the root directory (``\path\to\ml2dac``) of the project and run:

```
docker build -t ml2dac .
```

The process of building the image will take around 5 minutes.

#### Troubles Building the Docker image

**Permission Errors:**
In case you receive a permission error, you have to add the current user to the docker group:
```
sudo usermod -aG docker $USER
```

Instead of logging out/in again you can then run:

````
newgrp docker
````

**Timeout Error:** If the docker build command fails due to some Timeout error, you may have to add ``--network=host`` to the build command. So, you have to run:
```
docker build -t ml2dac . --network=host
``` 

**No Space Left on Device:** If you do not have any more space left on device due to docker containers, you can do some clean up by running 

```
docker system prune --all --force
```

### Run the Experiments

If the image is successfully built, you can run the container. 
Note that the default execution to fully reproduce the results takes around 72 hours.
Therefore, we provide some parameters to reduce the runtime or just change the experiment setting. 
This might have an effect on the results, but the trends should be the same as in the paper.
For instance, we only execute one run for each experiment per default.
We will execute the experiments for reproducing the figures 4 -7 and tables 3 -6 from the evaluation in our paper. 

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

In order to reduce runtime or only execute certain experiments, the following parameters are available: 

#### Additional Parameters

**Only Figures:**
This parameter sets if only the figures are generated, if set to `true` the other parameters become irrelevant. The default is `false`.

* `-e ONLY_FIGURES=true`: generates only the figures
* `-e ONLY_FIGURES=false`: experiments + figures

**Experiment:**
This parameter decides which experiments are executed. If all experiments should be executed, this parameter does not need to be defined.  The available values are RealWorld, Evaluation, VaryingTrainingData and Ablation. This can be passed as string but need to be comma separated: 

* `-e EXPERIMENTS=RealWorld,Evaluation` only the comparison on the Real World Data and Evaluation, which is the evaluation on synthetic data.
These are the most important experiments from our paper and the results are shown in the Figures 4 - 7. 
* `-e EXPERIMENTS=RealWorld,Evaluation,VaryingTrainingData,Ablation`: would execute all experiments 
* `-e EXPERIMENTS=RealWorld` just the comparison on the RealWorld Data 

**Runs:** The parameter `-e RUNS=1` defines the number of runs to execute for each experiment. 


A full example which executes only the experiments RealWorld and Evaluation with the parameters *warmstarts=1*,*runs=1* and *loops=1* would look like: 

```
docker run -e ONLY_FIGURES=false -e EXPERIMENTS=RealWorld,Evaluation -e WARMSTARTS=1 -e LOOPS=1 -e RUNS=1 -v local/dir/output:/app/evaluation_results/output  -d ml2dac
```