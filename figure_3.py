'''
(C) Copyright Siemens AG 2023

SPDX-License-Identifier: MIT
'''
import pickle

import numpy as np
import tqdm
from matplotlib import pyplot as plt
from numpy import random
from qiskit import Aer

from quantum_policy_evaluation import OneStepBanditQPE, n_samples_from_t, eps_bound


def sample_return(loose_prob_left, loose_prob_right, prob_left, horizon):
    """
    Sample one return classically
    """
    ret = 0
    for _ in range(horizon):
        action = "left" if random.rand(1) <= prob_left else "right"
        if action == "left":
            reward = 0 if random.rand(1) <= loose_prob_left else 1
        if action == "right":
            reward = 0 if random.rand(1) <= loose_prob_right else 1
        ret += reward
    return ret


# bandit parameters
loose_prob_left = 0.55
loose_prob_right = 0.65
# policy
prob_left = 0.5
# backend
backend = Aer.get_backend("statevector_simulator")
# n that determines number of samples
n_list = range(5, 11)
# how many runs per number of samples
n_runs = 1000

if __name__ == "__main__":
    qpe_dists = {}
    mc_dists = {}
    for n in n_list:
        n_samples = n_samples_from_t(n)
        # mote carlo
        mc_returns = []
        for _ in tqdm.tqdm(range(n_runs), desc="MC runs"):
            mc_return = 1 / n_samples * sum([
                sample_return(loose_prob_left, loose_prob_right, prob_left, horizon=1)
                for _ in range(n_samples)
            ])
            mc_returns.append(mc_return)
        mc_dists[n] = mc_returns
        # qpe
        qpe = OneStepBanditQPE(
            loose_prob_left=loose_prob_left,
            loose_prob_right=loose_prob_right,
            t_eval_qubits=n,
            backend=backend,
            cache_dir="./bandit_qpe_cache/1_rounds",
            verbose=True
        )
        dist = qpe.get_return_dist_exact(prob_left)
        print(dist)
        qpe_returns = random.choice(list(dist.keys()), size=n_runs, p=list(dist.values()))
        qpe_dists[n] = qpe_returns

    gt_ret = prob_left * (1 - loose_prob_left) + (1 - prob_left) * (1 - loose_prob_right)

    t_list = list(qpe_dists.keys())
    qpe_dists = list(qpe_dists.values())
    qpe_error_dists = [abs(qpe_dist - gt_ret) for qpe_dist in qpe_dists]
    qpe_error_means = [np.median(error_dist) for error_dist in qpe_error_dists]

    mc_dists = list(mc_dists.values())
    mc_error_dists = [abs(np.array(mc_dist) - gt_ret) for mc_dist in mc_dists]
    mc_error_means = [np.median(error_dist) for error_dist in mc_error_dists]

    fig, ax = plt.subplots(1, figsize=(6, 6))
    n_samples_list = [n_samples_from_t(t) for t in t_list]
    # MC
    ax.plot(n_samples_list, mc_error_means, color="orange", label="MC error median")
    ax.scatter(n_samples_list, mc_error_means, color="orange", marker="+")
    # QPE
    ax.plot(n_samples_list, qpe_error_means, color="blue", label="QPE error median")
    ax.scatter(n_samples_list, qpe_error_means, color="blue", marker="+")
    # eps bound
    ax.plot(n_samples_list, [eps_bound(t, 1, 0) for t in t_list], color="grey", label="QPE error bound")
    ax.scatter(n_samples_list, [eps_bound(t, 1, 0) for t in t_list], color="grey", marker="+")

    ax.legend(loc="best")
    ax.set_xlabel("Number of (q)samples")
    ax.set_ylabel("Median approximation error")
    plt.savefig("figure_3.pdf", bbox_inches="tight")
    plt.show()
