import os
from src.Experiments.Synthetic_Data import Evaluation
from src.Experiments.Real_world_experiment import Real_world_Experiment
from src.Experiments.Synthetic_Data.VaryingTrainingData import VaryTrainingData
from src.Experiments.Real_world_experiment.Ablation_Study import Ablation_Study
from evaluation_results.plots_notebooks.Papers_and_Figures import gen_figures



def main(only_figures='Default', experiments=['all'], loops=100, warmstarts=25, runs=10 ):

    if not only_figures:

        if('all' in experiments or 'evaluation' in experiments):
            print('evaluation experiment')

            print("evaluation done")
        

        if('all' in experiments or 'realworld' in experiments):
            print("real world experiment")
            print("real world experiment done")
            

        if('all' in experiments or 'varyingtrainingdata' in experiments):
            print("Varying Training Data")
            
            print("Varying training data done")
        

        if('All' in experiments or 'ablation' in experiments):
            print('ablation Study')
           
            print('Ablation study done ')
    
    print('Plots and figures')

    print('Plots and figures done')


if __name__ == '__main__':

    #mode -> default, figures, custom 
    # array of experiments: ['All',,'most important= Eval, RealWorld ''Evaluation, RealWorld, VaryingTrainingData, Ablation']

    #loops, warmstarts, runs

    only_figures = os.getenv('ONLY_FIGURES', 'false').lower() == 'true'    
    experiments = os.getenv('EXPERIMENTS', 'All').lower().split(',')
    #experiments = os.getenv('EXPERIMENTS', 'varyingtrainingdata').lower().split(',')

    loops = int(os.getenv('LOOPS', 100))
    warmstarts = int(os.getenv('WARMSTARTS', 25))
    runs = int(os.getenv('RUNS', 10))

    print('###################################')
    print(f'only figures = {only_figures}')
    print(f'experiments = {experiments}')
    print(f'loops = {loops}')
    print(f'warmstarts = {warmstarts}')
    print(f'runs = {runs}')
    print('###################################')
    main(only_figures=only_figures, experiments=experiments, loops=loops, warmstarts=warmstarts, runs=runs)