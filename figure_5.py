'''
(C) Copyright Siemens AG 2023

SPDX-License-Identifier: MIT
'''
import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from qiskit import Aer

from quantum_policy_evaluation import BanditQPEExperiment, eps_bound
from quantum_policy_iteration import BanditQuantumPolicyIterationExperiment

# bandit parameters
n_bandit_rounds = 1
loose_prob_left = 1
loose_prob_right = 0
# parameters for QPE
n_eval_qubits = 7
t_eval_qubits = n_eval_qubits + 3
backend = "statevector_simulator"
backend = Aer.get_backend(backend)
n_workers = 7
# parameters for quantum policy iteration
max_iter = 10000
patience = 30
lam = 8 / 7
# over how many searches to average
n_searches = 1000
# list of number of policies
# n_policies_list = [n ** 2 for n in range(40, 61)]
n_policies_list = [n ** 2 for n in range(40, 45)]

if __name__ == "__main__":
    # run for increasing number of policies and save quant_pol_iter_statistics to this
    quant_pol_iter_statistics = {}
    for n_policies in n_policies_list:
        # make stochastc policies
        prob_left_list = np.linspace(0, 1, n_policies)
        # if file does not exist, run qpe for all policies in prob_left_list and save output distributions
        toy_problem_qpe_file = f"./toy_problem_qpe_results/{n_bandit_rounds}_rounds/{t_eval_qubits}_{n_policies}.pkl"
        if os.path.exists(toy_problem_qpe_file):
            print(
                f"Found QPE distributions for toy problem with {n_policies} stochastic policies and t={t_eval_qubits}!\n"
                f"Reading them..."
            )
            with open(toy_problem_qpe_file, "rb") as f:
                qpe_experiment_results_dict = pickle.load(f)
        else:
            print(
                f"Did not find QPE distributions for toy problem with {n_policies} stochastic policies and t={t_eval_qubits}!\n"
                f"Making them..."
            )
            qpe_experiment = BanditQPEExperiment(
                n_bandit_rounds=1,
                loose_prob_left=loose_prob_left,
                loose_prob_right=loose_prob_right,
                t_eval_qubits=t_eval_qubits,
                backend=backend,
                cache_dir="./bandit_qpe_cache",
                verbose=True
            )
            qpe_experiment_results_dict = qpe_experiment.run_parallel(
                prob_left_list=prob_left_list,
                n_workers=n_workers,
                output_file=toy_problem_qpe_file
            )
        print("...done!")
        # qpe experiment quant_pol_iter_statistics to dataframes
        qpe_experiment_results = pd.DataFrame(qpe_experiment_results_dict)
        qpe_experiment_results.set_index("prob_left", inplace=True)
        # dataframe with qpe output distributions
        qpe_output_dists = [dct["qpe_ret_prob_dict"] for dct in qpe_experiment_results_dict]
        prob_lefts = [dct["prob_left"] for dct in qpe_experiment_results_dict]
        qpe_output_dists = pd.DataFrame(qpe_output_dists, index=prob_lefts)
        qpe_output_dists.fillna(0, inplace=True)
        qpe_output_dists = qpe_output_dists.div(qpe_output_dists.sum(axis=1), axis=0)  # normalize rows
        qpe_output_dists = qpe_output_dists.div(n_policies)  # normalize such that whole frame sums up to 1
        assert np.isclose(qpe_output_dists.values.sum(), 1)

        # setup policy iteration experiment
        print(f"Running {n_searches} quantum policy iteration searches with {n_workers} Pool workers...")
        quant_pol_iter = BanditQuantumPolicyIterationExperiment(
            qpe_dists=qpe_output_dists,
            max_iter=max_iter,
            patience=patience,
            lam=lam,
            verbose=False
        )
        # run n_searches runs of quantum policy iteration (always start with worst policy)
        searches = quant_pol_iter.run_parallel(
            init_prob_left_list=[1 for _ in range(n_searches)],
            init_pol_val_list=[0 for _ in range(n_searches)],
            n_workers=n_workers
        )
        print("...done!")
        # calculate statistics for searches
        n_success = 0
        successes = []
        n_iter_success = []
        n_rot_success = []
        for search in searches:
            prob_left = search["prob_left_list"][-1]
            gt_ret = qpe_experiment_results.loc[prob_left]["gt_ret"]
            # check
            if n_bandit_rounds == 1:
                successful = abs(gt_ret - 1) < eps_bound(n_eval_qubits, 1, 0)
            elif n_bandit_rounds == 2:
                successful = abs(gt_ret - 2) < eps_bound(n_eval_qubits, 2, 0)
            if successful == True:
                n_success += 1
                successes.append(search)
                n_iter_success.append(search["n_iter"] - patience)
                n_rot_success.append(sum(search["r_list"][0:-patience]))
        print(f"{n_success} / {n_searches} searches were successful.")
        print(
            f"On average, they needed {sum(n_iter_success) / n_searches} iterations and "
            f"{sum(n_rot_success) / n_searches} Grover rotations.\n",
        )
        mc_results = {
            "successes": successes,
            "n_success": n_success,
            "n_iter_success": n_iter_success,
            "n_rot_success": n_rot_success,
        }
        quant_pol_iter_statistics[n_policies] = mc_results

    # make plot
    fig, ax = plt.subplots(1, figsize=(12, 6))
    n_pol_list = list(quant_pol_iter_statistics.keys())
    x = [int(np.sqrt(n_pol)) for n_pol in n_pol_list]
    # boxplots
    plotargs = {
        "alpha": 0.4,
        "color": "black"
    }
    boxes = ax.boxplot(
        [quant_pol_iter_statistics[n_pol]["n_iter_success"] for n_pol in n_pol_list],
        positions=x,
        showmeans=True,
        meanprops={"marker": "o", "label": "mean", "markerfacecolor": "orange", "markeredgecolor": "orange"},
        boxprops=plotargs,
        # whis=0,
        whiskerprops=plotargs,
        medianprops=plotargs,
        capprops=plotargs,
        showfliers=False,
        flierprops={"alpha": 0.5},
    )
    # means and regression line
    means = [medline.get_ydata()[0] for medline in boxes["means"]]
    regression_line = np.poly1d(np.polyfit(x, means, 1))
    y_reg = regression_line(x)
    reg_plot, = ax.plot(x, y_reg, label="best linear fit", color="blue")
    mse = 1 / len(y_reg) * sum((np.array(means) - y_reg) ** 2)
    print(f"Regression MSE: {mse}")
    # layout and legend
    ax.set_ylabel("Grover rotations")
    ax.set_xlabel("number of policies (N)")
    ax.set_xticklabels([n_pol for n_pol in n_pol_list])
    plt.xticks(rotation=90)
    ax.legend(handles=[boxes["means"][0], reg_plot])
    plt.savefig("./figure_5.pdf", bbox_inches="tight")
