'''
(C) Copyright Siemens AG 2023

SPDX-License-Identifier: MIT
'''
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
from pathos.multiprocessing import Pool

from utils import verbose_print


class BanditQuantumPolicyIterationExperiment:

    def __init__(self, qpe_dists, max_iter, patience=None, lam=8 / 7, verbose=False):
        self.qpe_dists = qpe_dists
        self.max_iter = max_iter
        if patience is None:
            patience = max_iter
        self.patience = patience
        self.lam = lam
        self.verbose = verbose

    def print(self, txt):
        verbose_print(txt, self.verbose)

    def _grover_search(self, pol_val, n_rotations):
        """
        Grover search for QPI
        :param pol_val: estimate of policy value for oracle
        :param n_rotations: number of Grover rotations
        :return: the policy and the estimated value found by Grover search
        """
        # find states with higher values than thresh and normalize them
        success_states = self.qpe_dists[self.qpe_dists.columns[self.qpe_dists.columns > pol_val]]
        success_prob = success_states.values.sum()
        # find states with lower values than thresh and normalize them
        fail_states = self.qpe_dists[self.qpe_dists.columns[self.qpe_dists.columns <= pol_val]]
        # probability of improving or worsening via grover
        theta = np.arcsin(np.sqrt(success_prob))
        improv_prob = np.sin((2 * n_rotations + 1) * theta) ** 2
        worsen_prob = 1 - improv_prob
        # choose improving or worsening dataframe
        df = random.choices([success_states, fail_states], weights=[improv_prob, worsen_prob])[0]
        # sample row (policy) from that dataframe
        row = df.sample(weights=df.sum(axis=1))
        found_prob_left = row.index[0]
        # sample evaluation of that policy
        entry = row.sample(axis=1, weights=row.values[0])
        found_pol_val = entry.columns[0]
        return found_prob_left, found_pol_val

    def run(self, init_prob_left, init_pol_val):
        """
        Run quantum policy iteration
        :param init_prob_left: probability of initial policy that the left arm is chosen
        :param init_pol_val: estimate of value of initial policy
        :return: dict of type
             {
                "n_iter": number of iterations until the algorithm terminated (including patience)
                "prob_left_list": policies in all iterations
                "pol_val_list": estimated policy values for all iterations
                "r_list": Grover rotations for all iterations
                "m_list": m for all iterations
            }
        """
        # initialize parameters
        m = 1
        stat_count = 1
        n_iter = 0
        # record the run in this lists
        prob_left_list = [init_prob_left]
        pol_val_list = [init_pol_val]
        m_list = [m]
        r_list = [0]
        stat_count_list = [stat_count]
        # QPI iteration
        prob_left = init_prob_left
        pol_val = init_pol_val
        for i in range(1, self.max_iter):
            n_iter += 1
            if stat_count <= self.patience:
                # sample number of rotations from {0,...,ceil(m-1)}
                sample_range = range(0, int(np.ceil(m - 1) + 1))
                r = random.sample(sample_range, k=1)[0]
                # grover search
                found_prob_left, found_pol_val = self._grover_search(
                    pol_val=pol_val,
                    n_rotations=r
                )
                # check if improved
                if found_pol_val > pol_val:
                    self.print(f"Step {i}: Improved policy from {pol_val} to {found_pol_val} with {r} rotations.")
                    prob_left = found_prob_left
                    pol_val = found_pol_val
                    m = 1
                    stat_count = 1
                else:
                    m = self.lam * m
                    stat_count += 1
                # update lists
                prob_left_list.append(prob_left)
                pol_val_list.append(pol_val)
                r_list.append(r)
                m_list.append(m)
            else:
                self.print(f"Did not improve for {self.patience} iterations. Terminating.")
                break
        # make output
        search_record = {
            "n_iter": n_iter,
            "prob_left_list": prob_left_list,
            "pol_val_list": pol_val_list,
            "r_list": r_list,
            "m_list": m_list
        }
        self.print(f"Found policy with prob_left={prob_left_list[-1]} and estimated value {pol_val_list[-1]}\n")
        return search_record

    def run_parallel(self, init_prob_left_list, init_pol_val_list, n_workers):
        """
        Run quantum policy iteration for multiple initial conditions in parallel
        :param init_prob_left_list: prob_left values that parametrize initial policies
        :param init_pol_val_list: estimated values for initial policies
        :param n_workers: how many multiprocessing.Pool workers to use
        :return:
        """
        n_searches = len(init_prob_left_list)
        self.print(
            f"Running {n_searches} searches on {len(init_prob_left_list)} policies with {n_workers} Pool workers..."
        )
        start = datetime.now()
        pool = Pool(n_workers)
        search_records = pool.starmap(
            self.run,
            zip(init_prob_left_list, init_pol_val_list)
        )
        pool.close()
        pool.join()
        end = datetime.now()
        self.print(f"...done! Took {end - start}")
        return search_records


def load_toy_problem_qpe_output_dists(n_bandit_rounds, t_eval_qubits, n_policies):
    """
    Load output distributions of QPE for toy problem (1 or 2 rounds) with n_policies stochastic policies
    :param n_bandit_rounds:
    :param t_eval_qubits:
    :param n_policies:
    :return:
    """
    pkl_file = f"./toy_problem_qpe_results/{n_bandit_rounds}_rounds/{t_eval_qubits}_{n_policies}.pkl"
    with open(pkl_file, "rb") as f:
        ev = pickle.load(f)

    qpe_experiment_results = pd.DataFrame(ev)
    qpe_experiment_results.set_index("prob_left", inplace=True)

    qpe_output_dists = [dct["qpe_ret_prob_dict"] for dct in ev]
    prob_lefts = [dct["prob_left"] for dct in ev]
    qpe_output_dists = pd.DataFrame(qpe_output_dists, index=prob_lefts)
    qpe_output_dists.fillna(0, inplace=True)
    qpe_output_dists = qpe_output_dists.div(qpe_output_dists.sum(axis=1), axis=0)  # normalize rows
    qpe_output_dists = qpe_output_dists.div(n_policies)  # normalize such that whole frame sums up to 1
    assert np.isclose(qpe_output_dists.values.sum(), 1)
    return qpe_experiment_results, qpe_output_dists
