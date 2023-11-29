import json
import numpy as np
import os
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace
import ternary

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from subspace_algorithm.subspace_algorithm_utils.utils import (
    XY_from_csv,
    function_from_data,
)

from src.bax.util.misc_util import dict_to_namespace

from subspace_algorithm.subspace_algorithm_utils.metrics_evaluation import (
    EvaluationMetrics,
)
from subspace_algorithm.bax_subspace_algorithm_children import algorithm_type

from time import perf_counter




def plot(results_file: Path, config_path: Path, output_plot_fname: Path, acqfn_vis="SwitchBAX", fallback_acqfn="InfoBAX"):
    """

    Plots 3 plots: n_obtained vs Dataset Size ; Jaccard vs Dataset Size; Objective space sampling


    Args:
        results_file (Path): Path of consolidated pkl file
        config_path (Path): Path of config file used to generate pkl (saves memory)
        output_plot_fname (Path): output plot file name
        acqfn_vis (str, optional): The acqfn to visualize objective space. Defaults to "SwitchBAX".
        fallback_acqfn (str, optional): The fallback acqfn to visualize objective space. Defaults to "InfoBAX".
                                        Some np noise runs did not run SwitchBAX as the comparison was done between
                                        MeanBAX and InfoBAX

    Raises:
        ValueError: If no repeats are found
    """
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    config = dict_to_namespace(config)

    acquisition_strategy = (
        config.strategy
    )  # acquisition strategy. E.g. Run RS then run US
    n_fithyper = config.n_iter  # fit hparams at every iteration until n_fithyper
    strategy_list = [
        strat[0] for strat in acquisition_strategy
    ]  # Sampling strategy names (BAX, US etc..)

    dim_x = len(config.columns_x)
    dim_y = len(config.columns_y)
    x_full, y_full = XY_from_csv(
        config.dataset_path, config.columns_x, config.columns_y
    )

    fn = function_from_data(
        x_full, y_full
    )  # convert X, y discrete file to a callable function

    algo_params = {
        "name": config.subregion_algo_name,
        "subregion_algo_params": config.subregion_algo_params,
    }

    # Define subregion algorithm
    subregion_algo = algorithm_type(
        name=config.subregion_algo_name, algo_params=algo_params
    )

    desired_x_idx_true = subregion_algo.identify_subspace(x_full, y_full)

    with open(results_file, "rb") as pklfile:
        save_data_dict = pickle.load(pklfile)

    n_initial = 10
    n_desired = len(desired_x_idx_true)

    n_iter = n_iter = config.n_iter
    n_data = np.arange(n_initial, n_initial + n_iter)
    if n_iter <= n_desired:
        best_possible_n_obtained = np.arange(n_initial, n_iter + n_initial)
    elif n_desired <= n_initial:
        best_possible_n_obtained = n_desired * np.ones(n_iter)
    else:
        best_possible_n_obtained = list(np.arange(n_initial, n_desired)) + list(
            n_desired * np.ones(n_iter + n_initial - n_desired)
        )

    fig, axs = plt.subplots(1, 3, dpi=300, figsize=(16, 6))
    ax1 = axs[0]
    ax2 = axs[1]
    alpha = 0.2
    lw = 1.0
    for i, acqfn in enumerate(["InfoBAX", "MeanBAX", "RS", "US", "EHVI", "SwitchBAX"]):
        for result_metric in ["n_obtained", "jaccard_posterior"]:
            tmp_thing = save_data_dict[acqfn]
            nrpts = len(list(tmp_thing.keys()))
            rptkeys = list(tmp_thing.keys())

            if nrpts != 0:
                nitertmp = len(tmp_thing[rptkeys[0]][result_metric])
                data = np.zeros((nitertmp, nrpts))
                for iiter in range(0, nitertmp):
                    for irpt in range(0, nrpts):
                        data_obj = tmp_thing[rptkeys[irpt]]

                        data[iiter, irpt] = tmp_thing[rptkeys[irpt]][result_metric][
                            iiter
                        ]

                # data = results_dict[acqfn][result_metric]

                y = np.mean(data, axis=1)
                x = n_data
                if result_metric == "n_obtained":
                    ax = ax1
                if result_metric == "jaccard_posterior":
                    ax = ax2
                if acqfn == "InfoBAX":
                    ax.plot(x, y, label=acqfn, linewidth=lw, zorder=10)
                    ax.fill_between(
                        x,
                        np.clip(y - np.std(data, axis=1), 0.0, 1000000),
                        y + np.std(data, axis=1),
                        alpha=alpha,
                    )
                elif acqfn == "MeanBAX":
                    ax.plot(x, y, label=acqfn, linewidth=lw, zorder=9)
                    ax.fill_between(
                        x,
                        np.clip(y - np.std(data, axis=1), 0.0, 1000000),
                        y + np.std(data, axis=1),
                        alpha=alpha,
                    )
                elif acqfn == "SwitchBAX":
                    ax.plot(x, y, label=acqfn, linewidth=lw, zorder=11)
                    ax.fill_between(
                        x,
                        np.clip(y - np.std(data, axis=1), 0.0, 1000000),
                        y + np.std(data, axis=1),
                        alpha=alpha,
                    )
                else:
                    ax.plot(x, y, label=acqfn, linewidth=lw)
                    ax.fill_between(
                        x,
                        np.clip(y - np.std(data, axis=1), 0.0, 1000000),
                        y + np.std(data, axis=1),
                        alpha=alpha,
                    )

                ax1.set_xlabel("Dataset Size")
                ax2.set_xlabel("Dataset Size")

                if result_metric == "n_obtained" and i == 0:
                    ax1.plot(x, best_possible_n_obtained, "k--", label="Best Possible")
                    ax1.set_ylabel("Number Obtained")
                if result_metric == "jaccard_posterior" and i == 0:
                    ax2.plot(
                        x, np.ones(np.array(x).shape), "k--", label="Best Possible"
                    )
                    ax2.set_ylabel("Posterior Jaccard Index")

    ax3 = axs[2]
    if "ternary" in results_file.name:
        ax3.set_xlabel("Coercivity (mT)")
        ax3.set_ylabel("Kerr Rotation (mrad)")

    # Customize the plot
    elif "np" in results_file.name:
        ax3.set_xlabel("Nanoparticle Size (nm)")
        ax3.set_ylabel("Poly dispersity (%)")

    acqfn = acqfn_vis
    tmp_thing = save_data_dict[acqfn]
    nrpts = len(list(tmp_thing.keys()))
    rptkeys = list(tmp_thing.keys())

    if nrpts == 0:
        acqfn = fallback_acqfn
        tmp_thing = save_data_dict[acqfn]
        nrpts = len(list(tmp_thing.keys()))
        rptkeys = list(tmp_thing.keys())

    if nrpts != 0:
        collected_data_ids = save_data_dict[acqfn][rptkeys[0]]["collected_data_ids"]
        acqfns_data = ax3.scatter(
            y_full[:, 0], y_full[:, 1], c="#5F9AFF", s=30, alpha=0.4, label="All points"
        )

        ax3.scatter(
            y_full[desired_x_idx_true, 0],
            y_full[desired_x_idx_true, 1],
            c="#FFD864",
            s=30,
            linewidth=0.15,
            marker="D",
            edgecolor="k",
            label="Desired points",
        )
        ax3.scatter(
            y_full[collected_data_ids, 0],
            y_full[collected_data_ids, 1],
            alpha=1,
            s=8,
            linewidth=0.15,
            c="#DA0028",
            edgecolor="k",
            label="Measured points",
        )
    else:
        raise ValueError(f"{acqfn_vis} and {fallback_acqfn} has no repeats is pkl file")

    ax1.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.2))
    ax2.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.2))
    ax3.legend(loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.2))
    fig.tight_layout()
    fig.savefig(output_plot_fname)


if __name__ == "__main__":
    current_dir = Path(__file__).parent.resolve()
    results_path = current_dir.parent / "plotting/paper_pkls"
    plt.style.use(current_dir.parent /"subspace_algorithm/subspace_algorithm_utils/matplotlib.rc")

    config_list = ["configs/nanoparticle_synthesis/conditional_impossible.json",
                   "configs/nanoparticle_synthesis/conditional_possible.json",
                   "configs/nanoparticle_synthesis/library_noise_0.01_fixed_hypers.json",
                   "configs/nanoparticle_synthesis/library_noise_0.1.json",
                   "configs/nanoparticle_synthesis/library_noise_0.01.json",
                   "configs/nanoparticle_synthesis/library_noise_0.05.json",
                   "configs/nanoparticle_synthesis/library_noise_0.json",
                   "configs/nanoparticle_synthesis/percentile_union.json",
                   "configs/ternary_pd/multiband_fixed_hypers.json",
                   "configs/ternary_pd/multiband.json",
                   "configs/ternary_pd/pareto.json",
                   "configs/ternary_pd/wish_list_fixed_hypers.json",
                   "configs/ternary_pd/wish_list.json",
                   ]
    nconfigs = len(config_list)
    for iconfig in range(0, nconfigs):
        
        config_filepath = (
            current_dir.parent
            / config_list[iconfig]
        )
        label = config_filepath.name.rsplit(".", 1)[0]
        dset_full = config_filepath.parent.name
        if dset_full == "nanoparticle_synthesis":
            dset = "np"
        else:
            dset="ternary"

        results_file = results_path / "consolidated_results_{}_{}.pkl".format(dset, label)
       
        plot_path = current_dir.parent / "plot_visualizations"
        plot_path.mkdir(exist_ok=True)
        output_plot_fname = plot_path / "{}_{}.png".format(dset, label)
        plot(results_file, config_filepath, output_plot_fname)
