import os
from src.Experiments.Synthetic_Data import Evaluation
from src.Experiments.Real_world_experiment import Real_world_Experiment
from src.Experiments.Synthetic_Data.VaryingTrainingData import VaryTrainingData
from src.Experiments.Real_world_experiment.Ablation_Study import Ablation_Study
from evaluation_results.plots_notebooks.Papers_and_Figures import gen_figures


def main(only_figures='Default', experiments=['all'],
         loops=100,
         warmstarts=25,
         runs=10):
    if not only_figures:

        if ('all' in experiments or 'evaluation' in experiments):
            print('evaluation experiment')

            e_limit_cs = True
            e_cvi = 'predict'
            e_time_limit = 120 * 60
            e_mf_sets = ""
            mkr_path = "src/MetaKnowledgeRepository"
            Evaluation.run_experiment(limit_cs=e_limit_cs,
                                      cvi=e_cvi,
                                      n_warmstarts=warmstarts,
                                      n_loops=loops,
                                      time_limit=e_time_limit,
                                      mf_sets=e_mf_sets,
                                      mkr_path_string=mkr_path,
                                      result_folder="gen_results/evaluation_results/synthetic_data")

            print("evaluation done")

        if ('all' in experiments or 'realworld' in experiments):
            print("real world experiment")
            Real_world_Experiment.run_experiment(runs=runs, n_warmstarts=50, n_loops=loops)
            print("real world experiment done")

        if ('all' in experiments or 'varyingtrainingdata' in experiments):
            print("Varying Training Data")
            vt_limit_cs = True
            vt_result_path = "gen_results/evaluation_results/synthetic_data/vary_training_data"

            VaryTrainingData.run_experiment(n_warmstarts=warmstarts,
                                            n_loops=loops,
                                            limit_cs=vt_limit_cs,
                                            result_path=vt_result_path)

            print("Varying training data done")

        if ('all' in experiments or 'ablation' in experiments):
            print('ablation Study')
            a_components = ["all", "no_algo_reduction", "no_cvi_selection", "no_warmstart"]
            a_time_limit = 240 * 60
            Ablation_Study.run_experiment(runs=runs,
                                          n_warmstarts=50,
                                          n_loops=70,
                                          components=a_components,
                                          time_limit=a_time_limit)
            print('Ablation study done ')

    print('Plots and figures')

    if only_figures:
        gen_figures()
    else:
        fig_eval = 'all' in experiments or 'evaluation' in experiments
        fig_real_world = 'all' in experiments or 'realworld' in experiments
        fig_ablation = 'all' in experiments or 'ablation' in experiments
        fig_vary = 'all' in experiments or 'varyingtrainingdata' in experiments
        gen_figures(evaluation=fig_eval, real_world=fig_real_world, ablation=fig_ablation,
                    varying_training_data=fig_vary)

    print('Plots and figures done')


if __name__ == '__main__':
    only_figures = os.getenv('ONLY_FIGURES', 'false').lower() == 'true'
    experiments = os.getenv('EXPERIMENTS', 'All').lower().split(',')
    loops = int(os.getenv('LOOPS', 100))
    warmstarts = int(os.getenv('WARMSTARTS', 25))
    runs = int(os.getenv('RUNS', 1))

    print('###################################')
    print(f'only figures = {only_figures}')
    print(f'experiments = {experiments}')
    print(f'loops = {loops}')
    print(f'warmstarts = {warmstarts}')
    print(f'runs = {runs}')
    print('###################################')
    main(only_figures=only_figures, experiments=experiments, loops=loops, warmstarts=warmstarts, runs=runs)
