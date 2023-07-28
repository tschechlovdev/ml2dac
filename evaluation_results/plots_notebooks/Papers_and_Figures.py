import os
import pdb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.patheffects as path_effects
from pathlib import Path, PurePath
import warnings

from pandas.core.common import SettingWithCopyWarning

from Utils import Helper

warnings.filterwarnings(action="ignore",
                        # SettingWithCopyWarning
                        )


def add_median_labels(ax):
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[4:len(lines):lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y - 0.05 * y, f'{round(value)}',
                       horizontalalignment='center', fontsize=12, color='black', weight='semibold',
                       # bbox=dict(facecolor='#445A64')
                       # bbox=dict(facecolor='lightgray')
                       )


def add_avg_labels(ax):
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[5:len(lines):lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:.2f}', ha='center', va='center',
                       fontweight='bold', color='white', bbox=dict(facecolor='#445A69'))
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height() + 1
            value = '{:.0f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="center", fontsize="small")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)


def add_median_labels_2(ax):
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[4:len(lines):lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y - 0.3 * y, f'{round(value)}',
                       horizontalalignment='center', fontsize=14, color='black', weight='bold',
                       # bbox=dict(facecolor='#445A64')
                       # bbox=dict(facecolor='lightgray')
                       )
        # create median-colored border around white text for contrast
        # text.set_path_effects([
        #    path_effects.Stroke(linewidth=3, foreground=median.get_color()),
        #    path_effects.Normal(),
        # ])


def load_synthetic_data_results(evaluation=False):
    aml4c_df = pd.read_csv(f'evaluation_results/related_work/aml4c_results.csv', index_col=False)
    aml4c_df["type"] = aml4c_df["dataset"].apply(lambda x: x.split('-')[0].split('=')[1])
    aml4c_df["n"] = aml4c_df["dataset"].apply(lambda x: x.split('-')[2].split('=')[1]).astype(int)
    aml4c_df["d"] = aml4c_df["dataset"].apply(lambda x: x.split('-')[3].split('=')[1]).astype(int)
    aml4c_df["noise"] = aml4c_df["dataset"].apply(lambda x: x.split('-')[4].split('=')[1]).astype(float)
    aml4c_df["k"] = aml4c_df["dataset"].apply(lambda x: x.split('-')[1].split('=')[1]).astype(float)
    aml4c_df["ARI"] = aml4c_df["ARI"].apply(lambda x: x if x > 0 else x * -1)
    aml4c_df["Best ARI"] = aml4c_df["ARI"]
    aml4c_df["Method"] = aml4c_df.apply(lambda x: f"AML4C ({x['metric']})", axis=1)

    merged_results = pd.DataFrame()

    base_path = ''
    if evaluation:
        base_path = 'gen_results/'

    stats_general_df = pd.read_csv(f'{base_path}evaluation_results/synthetic_data/results_statistical+general.csv',
                                   index_col=None)
    # stats_df = pd.read_csv(f'../../synthetic_data/results_statistical.csv', index_col=None)
    # stats_general_df = pd.read_csv(f'statistical+general_online_result.csv', index_col =None)
    stats_info_general_df = pd.read_csv(
        f'{base_path}evaluation_results/synthetic_data/results_statistical+info-theory+general.csv', index_col=None)

    a_for_dc_autocluster_df = pd.read_csv(f'evaluation_results/synthetic_data/results_autocluster.csv', index_col=None)

    # stats_df["Method"] = "ML2DAC (Stats)"
    stats_general_df["Method"] = "ML2DAC (Stats+General)"
    stats_info_general_df["Method"] = "ML2DAC (Stats+Info+General)"
    a_for_dc_autocluster_df["Method"] = "ML2DAC - AutoCluster"

    # related_work_online_df = pd.read_csv('related_work/related_work_online_result.csv', index_col=None)
    related_work_online_df = pd.read_csv('evaluation_results/related_work/autoclust.csv', index_col=None)

    related_work_online_df["Method"] = "AutoClust"
    related_work_online_df["Best ARI"] = related_work_online_df["ARI"]

    autocluster_df = pd.read_csv("evaluation_results/related_work/autocluster_mv.csv")
    autocluster_df = autocluster_df[autocluster_df["Method"] == "AutoCluster - MV"]
    autocluster_df["Method"] = "AutoCluster"
    autocluster_df["Best ARI"] = autocluster_df["ARI"]
    merged_results = pd.concat([  # stats_df,
        stats_general_df, stats_info_general_df, a_for_dc_autocluster_df,
        autocluster_df, related_work_online_df, aml4c_df])

    merged_results = pd.concat([merged_results, related_work_online_df])
    related_work_online_df[related_work_online_df['ARI'] == related_work_online_df['ARI'].min()]
    merged_results = pd.concat([merged_results, aml4c_df])

    # Merge Results/ Some Preprocessing Stuff

    merged_results["ARI"] = merged_results["ARI"].apply(lambda x: x if x > 0 else x * -1)
    merged_results["Best ARI"] = merged_results["Best ARI"].apply(lambda x: x if x > 0 else x * -1)

    if "algorihtm" in merged_results.columns:
        merged_results["algorithm"] = merged_results["algorihtm"]
        merged_results = merged_results.drop("algorihtm", axis='columns')

    ############################################################################
    #### Some Preprocessing stuff --> assign dataset properties as columns #####

    # merged_results["time interval"] = merged_results["wallclock time"].apply(lambda x: int(x/30))
    merged_results["type"] = merged_results["dataset"].apply(lambda x: x.split('-')[0].split('=')[1])
    merged_results["n"] = merged_results["dataset"].apply(lambda x: x.split('-')[2].split('=')[1]).astype(int)
    merged_results["d"] = merged_results["dataset"].apply(lambda x: x.split('-')[3].split('=')[1]).astype(int)
    merged_results["noise"] = merged_results["dataset"].apply(lambda x: x.split('-')[4].split('=')[1]).astype(float)
    merged_results["k"] = merged_results["dataset"].apply(lambda x: x.split('-')[1].split('=')[1]).astype(float)
    return merged_results, aml4c_df


def generate_figure_4(evaluation, merged_results):
    methods_to_use = ["ML2DAC (Stats+General)",
                      "ML2DAC (Stats+Info+General)",
                      "AML4C (DBCV)",
                      "AML4C (COP)",
                      "AutoClust",
                      "AutoCluster",
                      ]
    selcted_methods = merged_results[merged_results["Method"].isin(methods_to_use)]
    selcted_methods["ARI"] = selcted_methods["ARI"] * 100
    selcted_methods["Best ARI"] = selcted_methods["Best ARI"] * 100

    selcted_methods["Method"] = pd.Categorical(selcted_methods["Method"], ordered=True, categories=methods_to_use)
    selcted_methods = selcted_methods.reset_index()

    if evaluation:
        ax = sns.lineplot(x="iteration", y="Best ARI", markevery=[0] + [x for x in range(4, 100, 5)] + [99],
                          markers=True,
                          style="Method",
                          data=selcted_methods
                          , hue="Method",
                          )
    else:
        ax = sns.lineplot(x="iteration", y="ARI", markevery=[0] + [x for x in range(4, 100, 5)] + [99], markers=True,
                          style="Method",
                          data=selcted_methods
                          , hue="Method",
                          )

    ax.set_xticks([0] + list(range(10, 105, 10)))
    ax.set_xlabel(r"Optimizer Loop")
    ax.set_ylim([0, 100])
    ax.set_yticks([x for x in range(0, 110, 10)])
    ax.set_ylabel("ARI (in %)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    legend = plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(bbox_to_anchor=(1.0, -0.2), loc=0, borderaxespad=0., ncol=3)
    ax.axvline(x=26, c='black', linestyle="--", linewidth=1, label="Warmstart Configs")
    ax.text(8, 101, "Warmstart Configurations")

    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("evaluation_results/output/Fig4_accuracy_average_baselines.pdf", bbox_inches='tight')
    # plt.show()
    plt.close()
    return selcted_methods


def generate_figure_5(evaluation, selcted_methods):
    methods_to_use = ["ML2DAC (Stats+General)",
                      "ML2DAC (Stats+Info+General)",
                      "AML4C (DBCV)",
                      "AML4C (COP)",
                      "AutoClust",
                      "AutoCluster",
                      ]

    font = {'family': 'DejaVu Sans',
            'weight': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    types = ["circles", "moons", "gaussian", "varied"]
    for type_, subfig in zip(types,
                             # selcted_methods["type"].unique(),
                             ["a", "b", "c", "d"]):
        type_methods = selcted_methods[selcted_methods["type"] == type_]
        # type_methods["ARI"] = type_methods["ARI"] * 100
        flierprops = dict(markerfacecolor='grey',
                          markersize=8,
                          # linestyle='--'
                          )

        data = type_methods[(type_methods["iteration"] == 25)
            # & (type_methods["noise"] < 0.1)
        ]
        ari_column = 'ARI'
        if evaluation: ari_column = 'Best ARI'
        avgs = data.groupby(['Method'])[ari_column].mean()
        medians = data.groupby(['Method'])[ari_column].median()
        medians = medians.reset_index()
        medians["Method"] = pd.Categorical(medians["Method"], ordered=True, categories=methods_to_use)
        medians = medians[ari_column].values
        # medians = medians.sort_values("Method")

        vertical_offset = data['ARI'].mean() * 0.05  # offset from median for display

        ax = sns.boxplot(data=data,
                         x="Method",
                         y=ari_column,
                         flierprops=flierprops,
                         )

        for xtick in ax.get_xticks():
            text = ax.text(xtick, medians[xtick] + 2, f'{round(medians[xtick])}',
                           horizontalalignment='center', fontsize=14, color='black', weight='bold',
                           )
        ax.set_xticklabels([], rotation=35, ha="right")
        # ax.set_title(type_)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(-5, 105)
        ax.set_ylabel(r'ARI (in %)')
        ax.set_yticks([x for x in range(0, 110, 10)])
        plt.savefig(f"evaluation_results/output/Fig_5{subfig}_accuracy_distribution_{type_}.pdf", bbox_inches='tight')
        # plt.show()
        plt.close()


def generate_figure_6(evaluation, aml4c_df):
    # Generate Figure 6a and 6b
    base_path = ''
    if evaluation:
        base_path = 'gen_results/'
    stats_general_df = pd.read_csv(f'{base_path}evaluation_results/synthetic_data/results_statistical+general.csv',
                                   index_col=None)
    # stats_df = pd.read_csv(f'../../synthetic_data/results_statistical.csv', index_col=None)
    # stats_general_df = pd.read_csv(f'statistical+general_online_result.csv', index_col =None)
    stats_info_general_df = pd.read_csv(
        f'{base_path}evaluation_results/synthetic_data/results_statistical+info-theory+general.csv', index_col=None)

    a_for_dc_autocluster_df = pd.read_csv(f'evaluation_results/synthetic_data/results_autocluster.csv', index_col=None)

    stats_general_df["Method"] = "ML2DAC (Stats+General)"
    stats_info_general_df["Method"] = "ML2DAC (Stats+Info+General)"
    a_for_dc_autocluster_df["Method"] = "ML2DAC - AutoCluster"

    related_work_online_df = pd.read_csv('evaluation_results/related_work/autoclust.csv', index_col=None)

    related_work_online_df["Method"] = "AutoClust"

    autocluster_df = pd.read_csv("evaluation_results/related_work/autocluster_mv.csv")
    autocluster_df["Method"] = "AutoCluster"
    merged_results = pd.concat([stats_general_df, stats_info_general_df, a_for_dc_autocluster_df,
                                autocluster_df, related_work_online_df, aml4c_df])

    merged_results = pd.concat([merged_results, related_work_online_df])
    related_work_online_df[related_work_online_df['ARI'] == related_work_online_df['ARI'].min()]
    merged_results = pd.concat([merged_results, aml4c_df])

    autoCluster_pivot = autocluster_df.pivot(index=["dataset", "iteration"], columns="metric", values=["runtime"])
    autoCluster_pivot["sum"] = autoCluster_pivot["runtime"]["CH"].cumsum() + autoCluster_pivot["runtime"][
        "SIL"].cumsum() + autoCluster_pivot["runtime"]["DBI"].cumsum()

    related_work_online_df["wallclock time"] = related_work_online_df.apply(
        lambda row: max(7200, row["max wallclock"]) if row["iteration"] >= row["max iteration"] else row[
            "wallclock time"], axis=1)

    mv_autocluster = autocluster_df
    for data, iteration in autoCluster_pivot.index:
        mv_autocluster.loc[(mv_autocluster["dataset"] == data) & (mv_autocluster["iteration"] == iteration)][
            "wallclock time"] = autoCluster_pivot["sum"][data, iteration]
    mv_autocluster["wallclock time"] = autoCluster_pivot["sum"].reset_index()["sum"]
    for data in mv_autocluster["dataset"].unique():
        mv_autocluster.loc[mv_autocluster["dataset"] == data, "max wallclock"] = \
            mv_autocluster[mv_autocluster["dataset"] == data]["wallclock time"].max()
    mv_autocluster["Method"] = "AutoCluster"
    mv_autocluster["wallclock time"].max()

    merged_df = pd.DataFrame()

    autocluster_df["Method"] = "AutoCluster"
    for df in [stats_general_df, stats_info_general_df, aml4c_df[aml4c_df["metric"] == "DBCV"],
               aml4c_df[aml4c_df["metric"] == "COP"], related_work_online_df, mv_autocluster
               ]:
        new_df = pd.DataFrame()
        for dataset in df["dataset"].unique():
            mask = (df["dataset"] == dataset)
            data_df = df[mask]
            # max_iteration = data_df["max iteration"].max()
            max_iteration = data_df["iteration"].max()
            # max_iteration = data_df["max iteration"].values[0]
            mask_2 = data_df["iteration"] <= max_iteration
            data_df = data_df[mask_2]

            data_df["wallclock time"] = data_df["runtime"].cumsum()

            data_df["max wallclock"] = data_df["wallclock time"].max()
            new_df = pd.concat([new_df, data_df])

        merged_df = pd.concat([merged_df, new_df])
    merged_df = pd.concat([merged_df, mv_autocluster, related_work_online_df])
    merged_df[(merged_df["Method"] == "AutoCluster")]["wallclock time"].max()

    merged_df["type"] = merged_df["dataset"].apply(lambda x: x.split('-')[0].split('=')[1])
    merged_df["n"] = merged_df["dataset"].apply(lambda x: x.split('-')[2].split('=')[1]).astype(int)
    merged_df["d"] = merged_df["dataset"].apply(lambda x: x.split('-')[3].split('=')[1]).astype(int)
    merged_df["noise"] = merged_df["dataset"].apply(lambda x: x.split('-')[4].split('=')[1]).astype(float)
    merged_df["k"] = merged_df["dataset"].apply(lambda x: x.split('-')[1].split('=')[1]).astype(float)
    if "algorihtm" in merged_df.columns:
        merged_df["algorithm"] = merged_df["algorihtm"]
        merged_df = merged_df.drop("algorihtm", axis='columns')
    afourC_cop = merged_df[merged_df["Method"] == "AutoCluster"]

    flierprops = dict(markerfacecolor='grey',
                      markersize=8,
                      # linestyle='--'
                      )

    data = merged_df[(merged_df["iteration"] == 100) & (merged_df["n"] == 10000)]
    avgs = data.groupby(['Method'])['max wallclock'].median()
    medians = data.groupby(['Method'])['max wallclock'].median()

    vertical_offset = {0: 30, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50}
    ax = sns.boxplot(data=data, x="Method", y="max wallclock", flierprops=flierprops
                     )
    ax.set_yticklabels([10, 100, 1000, 10000])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yscale("log")
    legend = plt.legend(bbox_to_anchor=(1.0, -0.2), loc=0, borderaxespad=0., ncol=3,
                        # handles=handles[1:], labels=labels[1:]
                        )
    add_median_labels(ax)
    ax.set_xticklabels([], rotation=35, ha="right")
    y_labels = ax.get_yticklabels()
    y_lims = ax.get_ylim()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel("Runtime (in Seconds)")
    plt.savefig("evaluation_results/output/Fig6b_runtime_comparison_large.pdf", bbox_inches='tight')
    plt.close()

    flierprops = dict(markerfacecolor='grey',
                      markersize=8,
                      # linestyle='--'
                      )

    data = merged_df
    avgs = data.groupby(['Method'])['max wallclock'].median()
    medians = data.groupby(['Method'])['max wallclock'].median()

    ax = sns.boxplot(data=merged_df, x="Method", y="max wallclock",
                     # patch_artist=True,
                     # hue="n",
                     # showmeans=True,
                     # meanprops={"marker":"X","markerfacecolor":"white", "markeredgecolor":"black", "markersize": 12},
                     # estimator=np.median
                     showfliers=False
                     )

    ax.set_xticklabels([], rotation=45, ha="right")
    ax.set_yscale("log")
    add_median_labels_2(ax)
    ax.set_yticklabels(y_labels)
    # ax.set_ylim(y_lims)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel("Runtime (in Seconds)")
    plt.savefig("evaluation_results/output/Fig6a_runtime_comparison.pdf", bbox_inches='tight')
    plt.close()


def generate_table_4(varying_training_data):
    if varying_training_data:
        path = 'gen_results/evaluation_results/synthetic_data/vary_training_data/run_0/'
        #path = f"evaluation_results/varying_training_data"
    else:
        path = f"evaluation_results/varying_training_data"

    varying_training_data_results = pd.DataFrame()
    for n_train in list(range(2, 72, 10)):
        dataset_results = os.listdir(path + f"/n_training_data_{n_train}")
        results = pd.concat(
            [pd.read_csv(path + f"/n_training_data_{n_train}" + f"/{data}") for data in dataset_results],
            ignore_index=True)
        results["#Training Datasets"] = n_train
        results["Best ARI"] *= -1
        results["ARI"] = results["Best ARI"]
        varying_training_data_results = pd.concat([varying_training_data_results, results], ignore_index=True)

    pd.DataFrame(varying_training_data_results[(varying_training_data_results["iteration"] == 100) & (
        varying_training_data_results["#Training Datasets"].isin(list(range(2, 64, 10))))].groupby(
        "#Training Datasets")[
                     "ARI"].mean().round(4)).transpose().to_csv(
        "evaluation_results/output/Table4.csv")


def generate_figure_7(real_world):
    data_path = Path("real_world_data/")
    datasets = [f for f in os.listdir(data_path) if os.path.isfile(data_path / f)]

    if real_world:
        ml2dac_path = Path("gen_results/evaluation_results/real_world/ML2DAC/")
    else:
        ml2dac_path = Path("evaluation_results/real_world/ML2DAC/")

    ml2dac_mf_paths = [Path(x[0]) for x in os.walk(ml2dac_path)]

    if real_world:
        aml4c_path = Path("gen_results/evaluation_results/real_world/Baselines/AML4C/")
    else:
        aml4c_path = Path("evaluation_results/real_world/Baselines/AML4C/")

    all_results = pd.DataFrame()

    replacements = {"gen_results/": "", "evaluation_results/": "", "\\": "-", "/": "-", "-warmstarts": "", "..-": "",
                    "real_world-": "", "Baselines-": "",
                    "statistical": "Stats",
                    "general": "General", "info-theory": "Info", "-similar_datasets": "",
                    "-n_similar_1": "",
                    "-run_0": "",
                    "-abl_study": "",
                    "_25": "",
                    "-all": "",
                    "-warmstart_and_bo": "",
                    "evaluation_results-": "",
                    "gen_results-": ""}

    aml4c_cvi_paths = [Path(x[0]) for x in os.walk(aml4c_path)]
    autoclust_path = Path("evaluation_results/real_world/Baselines/AutoClust/")
    autocluster_path = Path("evaluation_results/real_world/Baselines/AutoCluster/MV")
    method_paths = [autoclust_path, autocluster_path]
    method_paths.extend(aml4c_cvi_paths)
    method_paths.extend(ml2dac_mf_paths)
    for data in datasets:
        for method_path in method_paths:
            method_name = str(PurePath(method_path))
            for to_replace, replacement in replacements.items():
                method_name = method_name.replace(to_replace, replacement)

            data_path = method_path / data

            if data_path.exists():

                df = pd.read_csv(data_path)
                if len(df) > 0:
                    df["Method"] = method_name
                    df["dataset"] = data

                    df = df.sort_values("iteration", ascending=True)

                    if "Best ARI" not in df.columns:
                        df["Best CVI"] = df["metric score"].cummin()
                        df["Best ARI"] = 0
                        df.loc[df["iteration"] == 0, "Best ARI"] = df["ARI"].values[0]
                        df["changed"] = df["Best CVI"].diff()
                        df["Best ARI"] = df.apply(lambda x: x["ARI"] if x["changed"] > 0 else x["Best ARI"], axis=1)

                    df["Best ARI"] = df["Best ARI"] * -1
                    df["Best* ARI"] = df["Best ARI"].cummax()
                    df["ARI"] = df["ARI"] * -1
                    df["Ground-truth Best ARI"] = df["ARI"].cummax()

                    all_results = pd.concat([all_results, df])
            else:
                pass
        # autoclust_data_df = pd.read_csv(autoclust_path / data)

    all_results = all_results.reset_index(drop=True)
    all_results["Method"].unique()

    methods = ["ML2DAC-Stats+General",
               "ML2DAC-Stats+Info+General",
               "AML4C-DBCV",
               "AML4C-COP",
               "AutoClust",
               "AutoCluster-MV",
               ]

    method_results = all_results[(all_results["Method"].isin(methods))]
    # exit()
    # Generate Figure 7
    plt.close()
    plt.figure()
    iterations = list(range(0, 101, 5))
    iterations.append(1)
    # iterations.remove(0)
    method_results["Method"] = pd.Categorical(method_results["Method"], ordered=True, categories=methods)
    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')

    data = method_results[method_results["iteration"].isin(iterations)]
    data = data[~data["dataset"].isin(["mnist-fashion.csv", "mnist-digits.csv",
                                       # "letter.csv",
                                       "handwritten_digits.csv"
                                       ])]
    data["Best ARI"] = data["Best ARI"] * 100
    # data = data[~data["dataset"].isin(bad_data)]
    ax = sns.lineplot(x="iteration", y="Best ARI",
                      data=data,
                      hue="Method",
                      style="Method", markers=True, markevery=1,  # ci=None
                      )
    # ax.set_title("Average ARI over all real-world datasets")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("ARI (in %)")
    ax.set_xlabel("Optimizer Loop")
    ax.set_yticks(range(0, 46, 5))
    iterations = list(range(0, 101, 10))
    plt.xticks(iterations)
    legend = plt.legend(bbox_to_anchor=(1.0, -0.2), loc=0, borderaxespad=0., ncol=3,
                        # handles=handles[1:], labels=labels[1:]
                        )
    ax.axvline(x=51, c='black', linestyle="--", linewidth=1, label="Warmstart Configs")
    ax.text(30, 55, "Warmstart Configurations")

    plt.savefig("evaluation_results/output/Fig7_real_world_data_accuracy_comparison.pdf", bbox_inches='tight')
    return method_results


def generate_table_5(real_world, method_results):
    method_results["Runtime (s)"] = method_results["wallclock time"]

    mean_runtime = method_results[method_results["iteration"] == 100].groupby(["Method"])[
        "Runtime (s)"].mean().round(2) .reset_index()

    median_runtime = method_results[method_results["iteration"] == 100].groupby(["Method"])[
        "Runtime (s)"].median().round(2) .reset_index()

    runtime_real_world_df = mean_runtime.join(median_runtime.set_index("Method"), on="Method", lsuffix=" Median",
                                              rsuffix=" Mean")

    runtime_real_world_df.to_csv("evaluation_results/output/Table5.csv")


def generate_table_6(ablation):
    if ablation:
        path_to_abl_study_results = Path("gen_results/evaluation_results/real_world/abl_study/statistical+general/")
        #path_to_abl_study_results = Path("evaluation_results/abl_study/statistical+general")
    else:
        path_to_abl_study_results = Path("evaluation_results/abl_study/statistical+general/")

    components = os.listdir(path_to_abl_study_results)

    # components = os.listdir("abl_study/statistical+general/warmstarts_50")

    abl_results = pd.DataFrame()
    for component in components:
        runs = os.listdir(path_to_abl_study_results / component)
        for run in runs:
            data_results = pd.concat([pd.read_csv(path_to_abl_study_results / component / run / dataset) for dataset in
                                      os.listdir(path_to_abl_study_results / component / run)])
            data_results["run"] = run
            data_results["component"] = component
            abl_results = pd.concat([abl_results, data_results])

    abl_results["ARI"] = abl_results["Best ARI"]
    abl_results = abl_results.drop("Best ARI", axis=1)
    abl_results = abl_results[abl_results["iteration"] == 70].groupby("component")["ARI"].mean().round(4) * -1

    abl_results.to_csv("evaluation_results/output/Table6.csv")


def generate_table_3(evaluation=False):
    merged_results = pd.DataFrame()
    mfs = [
        "general",
        "statistical",
        "info-theory",
        ["statistical", "info-theory"],
        ["statistical", "general"],
        ["info-theory", "general"],
        ["statistical", "info-theory", "general"],
        # "optics",
        "meanshift",
        "autocluster"
    ]

    mf_mapping = {
        "general": "General",
        "statistical": "Stats",
        "info-theory": "Info",
        "+".join(["statistical", "info-theory"]): "Stats+Info",
        "+".join(["statistical", "general"]): "Stats+General",
        "+".join(["info-theory", "general"]): "Info+General",
        "+".join(["statistical", "info-theory", "general"]): "Stats+Info+General",
        # "optics": "O",
        "meanshift": "MF - AutoClust",
        "autocluster": "MF - AutoCluster",
        #    "CVI (CH)": "CVI (CH)"
    }

    mf_orders = ['+'.join(x) if isinstance(x, list) else x for x in mfs]
    print(mf_orders)
    mf_results = pd.DataFrame()

    used_mf_sets = ["statistical+info-theory+general", "statistical+general"]
    for mf in mfs:
        if isinstance(mf, list):
            mf = '+'.join(mf)

        if evaluation and (mf == "statistical+info-theory+general" or mf == "statistical+general"):
            mf_df = pd.read_csv(f"gen_results/evaluation_results/synthetic_data/results_{mf}.csv")
        else:
            import os
            mf_df = pd.read_csv(f"evaluation_results/synthetic_data/results_{mf}.csv")
            mf_df["Best ARI"] = mf_df["ARI"]
        # for x in ["_1", "_2", "_3"]:
        #    mf_df = pd.read_csv(f"{mf}_online_result{x}.csv", index_col=0)
        mf_df["Meta-Feature Set"] = mf
        mf_results = pd.concat([mf_results, mf_df])

    mf_results["Meta-Feature Set"].unique()
    mf_results["ARI"] = mf_results["ARI"].apply(lambda x: x if x > 0 else x * -1)
    mf_results["Best ARI"] = mf_results["Best ARI"].apply(lambda x: x if x > 0 else x * -1)

    mf_results["Meta-Feature Set"] = pd.Categorical(mf_results["Meta-Feature Set"], categories=mf_orders, ordered=True)
    mf_results["Meta-Feature Set"] = mf_results["Meta-Feature Set"].apply(lambda mf: mf_mapping[mf])
    mf_results = mf_results

    ############################################################################
    #### Some Preprocessing stuff --> assign dataset properties as columns #####

    mf_results["type"] = mf_results["dataset"].apply(lambda x: x.split('-')[0].split('=')[1])
    mf_results["n"] = mf_results["dataset"].apply(lambda x: x.split('-')[2].split('=')[1]).astype(int)
    mf_results["d"] = mf_results["dataset"].apply(lambda x: x.split('-')[3].split('=')[1]).astype(int)
    mf_results["noise"] = mf_results["dataset"].apply(lambda x: x.split('-')[4].split('=')[1]).astype(float)
    mf_results["k"] = mf_results["dataset"].apply(lambda x: x.split('-')[1].split('=')[1]).astype(float)
    mf_results["Meta-Feature Set"].unique()

    metric_ranking = pd.read_csv("evaluation_results/synthetic_data/metric_ranking.csv", index_col=0)
    metric_ranking["accuracy"] = metric_ranking.apply(lambda x: int(x["metric"] == x["optimal metric"]), axis=1)
    metric_ranking["Meta-Feature Set"] = metric_ranking["mf_set"]
    metric_ranking = metric_ranking.drop("mf_set", axis=1)
    metric_ranking["Meta-Feature Set"] = metric_ranking["Meta-Feature Set"].apply(
        lambda mf: mf if mf == "optics" else mf_mapping[mf])
    cvi_acc = metric_ranking.groupby(["Meta-Feature Set"])["accuracy"].mean().reset_index()
    cvi_acc = cvi_acc[cvi_acc["Meta-Feature Set"] != "optics"]
    mf_ari_runtime = mf_results[mf_results["iteration"] == 25].groupby(["Meta-Feature Set"])[
        "Best ARI", "mf time"].mean().round(4).reset_index()

    mf_ari_runtime["CVI Selection"] = mf_ari_runtime["Meta-Feature Set"].apply(
        lambda x: Helper.cvi_accuracy(cvi_acc, x))
    mf_ari_runtime["ARI (w=25)"] = mf_ari_runtime["Best ARI"]
    mf_ari_runtime["Runtime Meta-Feature Extraction"] = mf_ari_runtime["mf time"]
    mf_ari_runtime = mf_ari_runtime.drop(["Best ARI", "mf time"], axis=1)

    # Generate Table 3
    output_path = Path("evaluation_results/output")
    output_path.mkdir(exist_ok=True, parents=True)
    mf_ari_runtime.to_csv("evaluation_results/output/Table3.csv")


def gen_figures(evaluation=False, real_world=False, ablation=False, varying_training_data=False):
    # Load Results for different Meta-Feature Sets (cf. Section 7.2)

    print("Start generating table 3 ...")
    generate_table_3(evaluation)
    print("Finished Generating Table 3")
    # Load Synthetic Data Results

    print("Loading results for synthetic data comparison ...")
    merged_results, aml4c_df = load_synthetic_data_results(evaluation)
    print("Finished")

    # Generate Figure 4
    print("Start generating Figure 4 ...")
    selcted_methods = generate_figure_4(evaluation, merged_results)
    print("Finished generating Figure 4")

    # Generate Figure 5 (a - d)
    print("Start generating Figure 5 ...")
    generate_figure_5(evaluation, selcted_methods)
    print("Finished generating Figure 5")

    # Generate Figure 6 (a and b)
    print("Start generating Figure 6 ...")
    generate_figure_6(evaluation, aml4c_df)
    print("Finished generating Figure 6")

    # Generate Table 4
    print("Start generating Table 4 ...")
    generate_table_4(varying_training_data)
    print("Finished Generating Table 4")

    # Generate Figure 7
    print("Start generating Figure 7 ...")
    method_results = generate_figure_7(real_world)
    print("Finished generating Figure 7")

    # Generate Table 5
    print("Start generating Table 5 ...")
    generate_table_5(real_world, method_results)
    print("Finished generating Table 5")

    # Generate Table 6 (Ablation Study)
    print("Start generating Table 6 ...")
    generate_table_6(ablation)
    print("Finished generating Table 6")


if __name__ == "__main__":
    gen_figures(evaluation=True,
                real_world=True,
                ablation=True,
                varying_training_data=True
                )
