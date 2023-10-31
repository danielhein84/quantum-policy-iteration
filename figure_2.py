'''
(C) Copyright Siemens AG 2023

SPDX-License-Identifier: MIT
'''
import pickle

import numpy as np
from matplotlib import pyplot as plt
from qiskit import Aer

from quantum_policy_evaluation import BanditQPEExperiment, eps_bound

# dynamics of bandit
loose_prob_left = 0.55
loose_prob_right = 0.65
# policy
prob_left = 0.5
# parameters for QPE
eps = 0.025
n_eval_qubits = 7
t_eval_qubits = n_eval_qubits + 4
backend = "statevector_simulator"
backend = Aer.get_backend(backend)

if __name__ == "__main__":
    # run qpe for policy
    experiment = BanditQPEExperiment(
        n_bandit_rounds=2,
        loose_prob_left=loose_prob_left,
        loose_prob_right=loose_prob_right,
        t_eval_qubits=t_eval_qubits,
        backend=backend,
        cache_dir="./bandit_qpe_cache",
        verbose=True
    )
    qpe_results = experiment.run_parallel(
        prob_left_list=[prob_left],
        n_workers=1,
    )

    # make plot
    qpe_ret_prob_dict = qpe_results[0]["qpe_ret_prob_dict"]  # dict of type {return: probability}
    # dict to bins
    n_bins = 2000
    bins = np.linspace(0, 2, n_bins + 1)
    bin_centers = [left + (bins[k + 1] - left) / 2 for k, left in enumerate(bins[:-1])]
    binned = dict(zip(
        bin_centers,
        np.zeros(len(bins[:-1]))
    ))
    for k, left in enumerate(bins[:-1]):
        right = bins[k + 1]
        center = left + (right - left) / 2
        for ret, prob in qpe_ret_prob_dict.items():
            if left <= ret < right:
                binned[center] += prob
    # actual plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(binned.keys(), binned.values(), align="center", width=bins[1], color="blue")
    # vertical lines for ground truth return and epsilon bound
    gt_ret = qpe_results[0]["gt_ret"]
    ax.axvline(gt_ret, linestyle="-", color="orange", label="true value")
    ax.axvline(gt_ret - eps, linestyle="--", color="black", label="epsilon bound")
    ax.axvline(gt_ret + eps, color="black", linestyle="--")
    # legend and layout
    ax.legend(loc="best")
    ax.set_xlim((gt_ret - 0.05, gt_ret + 0.05))
    ax.set_xlabel("QPE output")
    ax.set_ylabel("probability")
    plt.savefig("figure_2.pdf")
    plt.show()

    # calculate and print bound violation probability with actual epsilon
    eps = eps_bound(n_eval_qubits, 2, 0)
    oob_prob = 0
    for ret, prob in qpe_ret_prob_dict.items():
        if abs(gt_ret - ret) > eps:
            oob_prob += prob
    print(f"P(|x-gt_ret| <= epsilon) = {np.round(1 - oob_prob, 5)}")
