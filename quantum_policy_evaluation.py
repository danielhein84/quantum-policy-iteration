'''
(C) Copyright Siemens AG 2023

SPDX-License-Identifier: MIT
'''
import os
import pickle
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit.algorithms import EstimationProblem, AmplitudeEstimation
from qiskit.circuit import Parameter
from qiskit.circuit.library import MCMT, ZGate
from qiskit.circuit.library import PhaseEstimation
from qiskit_aer.library.save_instructions.save_probabilities import SaveProbabilitiesDict

from utils import bin_float_to_dec_float, verbose_print


def eps_bound(n, upper, lower):
    """
    Get QPE error bound according to Corollary 1
    """
    pi = np.pi
    try:
        summand1 = pi / (2 ** (n + 1))
    except OverflowError:
        summand1 = 0
    try:
        summand2 = (pi ** 2) / (2 ** (2 * n + 2))
    except OverflowError:
        summand2 = 0
    return (upper - lower) * (summand1 + summand2)


def n_samples_from_t(t):
    """
    Translate QPE parameter t to number of qsamples
    """
    return 2 * (sum([2 ** k for k in range(t)])) + 1


class BanditQPE:
    """
    Base class for the QPE algorithm of two-armed bandit MDPs.
    """

    def __init__(self, loose_prob_left, loose_prob_right, t_eval_qubits, backend, cache_dir="./bandit_qpe_cache",
                 verbose=False):
        """
        :param loose_prob_left: probability of winning 0$ when pulling the left arm
        :param loose_prob_right: probability of winning 1$ when pulling right arm
        :param t_eval_qubits: n for QPE algorithm
        :param backend: qiskit backend on which QPE is run
        :param cache_dir: directory in which to store and search for cached circuits
        :param verbose: set to True if output should be printed to the console
        """
        self.verbose = verbose
        self.print("Setting up QPE algorithm for bandit...")
        start = datetime.now()
        self.loose_prob_left = loose_prob_left
        self.loose_prob_right = loose_prob_right
        self.t_eval_qubits = t_eval_qubits  # int(n_eval_qubits + np.ceil(np.log2(1/(2*delta) + 1/2)))
        self.backend = backend
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.print("...getting environment gate...")
        self._init_environment_circ()
        self.print("...getting policy gate...")
        self._init_var_policy_circ()
        self.print("...getting MDP gate...")
        self._init_var_mdp_circ()
        self.print("...getting state preparation for Grover...")
        self._init_var_a_phi_circ()
        self.print("...getting Grover gate...")
        self._init_var_grover_circ()
        self.print("...getting QPE circuit...")
        self._init_var_qpe_circ()
        stop = datetime.now()
        self.print(f"...setup done! Total time: {stop - start}")

    def print(self, txt):
        verbose_print(txt, self.verbose)

    def _init_environment_circ(self):
        """
        Initialize environment gate
        """
        # environment gate
        env_circ = QuantumCircuit(2, name="E")
        env_circ.x(0)
        env_circ.cry(2 * np.arccos(np.sqrt(self.loose_prob_left)), 0, 1)
        env_circ.x(0)
        env_circ.cry(2 * np.arccos(np.sqrt(self.loose_prob_right)), 0, 1)
        self.env_circ = env_circ

    def _init_var_policy_circ(self):
        """
        Initialize variational policy gate with free parameter for probability of pulling the left arm
        """
        pol_circ = QuantumCircuit(1, name="Pi")
        self._prob_left_angle_param = Parameter("prob_left_angle")
        pol_circ.ry(self._prob_left_angle_param, 0)
        self.var_pol_circ = pol_circ

    def _init_var_mdp_circ(self):
        """
        Initialize variational the MDP gate with free parameter for policy
        """
        self.var_mdp_circ = None
        raise NotImplementedError

    def _init_return_circ(self):
        """
        Initialize return gate
        """
        self.return_circ = None
        raise NotImplementedError

    def _init_var_a_phi_circ(self):
        """
        Initialize variational the A_phi gate with free parameter for policy
        """
        self.var_a_phi_circ = None
        raise NotImplementedError

    @staticmethod
    def phi_inv(x):
        """
        phi^{-1} for last step of QPE algorithm.
        """
        raise NotImplementedError

    def _init_var_grover_circ(self):
        """
        Initialize variational the Grover gate with free parameter for policy.
        """
        qreg = QuantumRegister(self.var_a_phi_circ.num_qubits, name="g")
        var_grov_circ = QuantumCircuit(qreg, name="Grov")
        # z oracle
        var_grov_circ.x(qreg[-1])
        var_grov_circ.z(qreg[-1])
        var_grov_circ.x(qreg[-1])
        # A_phi^dagger
        a_phi_dg_circ = self.var_a_phi_circ.inverse()
        var_grov_circ = var_grov_circ.compose(a_phi_dg_circ, var_grov_circ.qubits)
        # S_0 gate
        for qubit in qreg:
            var_grov_circ.x(qubit)
        mcz = MCMT(ZGate(), len(qreg) - 1, 1)
        var_grov_circ = var_grov_circ.compose(mcz, var_grov_circ.qubits)
        for qubit in qreg:
            var_grov_circ.x(qubit)
        # A_phi
        var_grov_circ = var_grov_circ.compose(self.var_a_phi_circ, var_grov_circ.qubits)
        var_grov_circ = transpile(var_grov_circ, self.backend)
        self.var_grov_circ = var_grov_circ

    def _init_var_qpe_circ(self):
        """
        Initialize variational phase estimation circuit with free prob_left parameter for policy.
        """
        # check if we var_qpe_circ for this instance is in cache_dir
        qpe_circ_id = f"{self.loose_prob_left}_{self.loose_prob_right}_{self.t_eval_qubits}_{self.backend}"
        path = f"{self.cache_dir}/{qpe_circ_id}.pkl"
        if os.path.exists(path):
            self.print("...found existing QPE circuit! Reading it...")
            with open(path, "rb") as f:
                self.var_qpe_circ = pickle.load(f)
        else:
            # first make phase estimation
            var_phase_est_circ = PhaseEstimation(
                num_evaluation_qubits=self.t_eval_qubits,
                unitary=self.var_grov_circ
            )
            problem = EstimationProblem(
                state_preparation=self.var_a_phi_circ,
                objective_qubits=[-1],
                grover_operator=self.var_a_phi_circ
            )
            amplitude_estimation = AmplitudeEstimation(
                num_eval_qubits=self.t_eval_qubits,
                phase_estimation_circuit=var_phase_est_circ
            )
            var_qpe_circ = amplitude_estimation.construct_circuit(problem)
            # transpile
            self.print(f"...transpiling QPE circuit for {self.backend}...")
            self.bak_qpe_circ = var_qpe_circ
            var_qpe_circ = transpile(var_qpe_circ, self.backend)
            self.var_qpe_circ = var_qpe_circ
            # save var_qpe_circ in cache_dir
            with open(path, "wb") as f:
                pickle.dump(var_qpe_circ, f)

    def get_qpe_circuit(self, prob_left, measure=False):
        """
        Bind prob_left parameter of policy to phase estimation circuit.
        :param prob_left: probability with which the left arm is chosen
        :param measure: set to True if the n evaluation qubits should be measured
        :return: phase estimation circuit for QPE with bound
        """
        prob_left_angle = 2 * np.arccos(np.sqrt(prob_left))
        prob_left_angle_param = self.var_qpe_circ.parameters[0]
        qpe_circ = self.var_qpe_circ.bind_parameters({prob_left_angle_param: prob_left_angle})
        if measure:
            eval_creg = ClassicalRegister(self.t_eval_qubits, name="eval_creg")
            qpe_circ.add_register(eval_creg)
            qpe_circ.measure(qpe_circ.qubits[0:self.t_eval_qubits], eval_creg)
        return qpe_circ

    def _bitstr_phase_dict_to_value_dict(self, dct):
        """
        Helper function for below to convert the bitstring-encoded phases that phase estimation outputs to the values.
        :param dct: dictionary of type {phase_as_bitstring: probability}
        :return: dictionary of type {value: probability}
        """
        rets = []
        vals = []
        for bitstr_phase, val in dct.items():
            # if dict key is hex convert to bin
            print(bitstr_phase)
            
            bitstr_phase = bin(bitstr_phase)[2:]
            bitstr_phase = bitstr_phase[::-1]
            
            int(bitstr_phase, 2)
            # bitstr phase to decimal phase
            float_phase = bin_float_to_dec_float(f"0.{bitstr_phase}")
            # float phase to return
            ret = self.phi_inv(np.sin(np.pi * float_phase) ** 2)
            if ret in rets:
                vals[rets.index(ret)] += val
            else:
                rets.append(ret)
                vals.append(val)
        # make dict and sort descending
        out = dict(sorted(dict(zip(rets, vals)).items(), key=lambda k: k[1], reverse=True))
        return out

    def get_return_dist_exact(self, prob_left):
        """
        Get analytically calculated distribution of values returned by QPE for policy with prob_left.
        :param prob_left: probability with which the left arm is chosen
        :return: dictionary of type {value: probability}
        """
        qpe_circ = self.get_qpe_circuit(prob_left, measure=False)
        instr = SaveProbabilitiesDict(self.t_eval_qubits, label="probs")
        qpe_circ.append(instr, qpe_circ.qubits[0:self.t_eval_qubits])
        result = self.backend.run(qpe_circ, shots=1).result()  # qpe_circ is already transpiled
        probs = result.data()["probs"]
        return self._bitstr_phase_dict_to_value_dict(probs)

    def get_gt_return(self, prob_left):
        """
        Calculate exact ground truth value (return) of a policy.
        :param prob_left: probability with which the left arm is chosen
        :return: exact ground truth value
        """
        raise NotImplementedError


class ReducedOneStepBanditQPE(BanditQPE):
    """
    QPE for one step of two-armed bandit. The class is called "reduced" because in this case we can express the MDP
    model with returns as a single Ry gate.
    """

    def __init__(self, loose_prob_left, loose_prob_right, t_eval_qubits, backend, cache_dir, verbose):
        super().__init__(loose_prob_left, loose_prob_right, t_eval_qubits, backend, cache_dir, verbose)

    def _init_var_mdp_circ(self):
        # do not need policy or environment gates
        self.var_pol_circ = None
        self.env_circ = None
        var_mdp_circ = QuantumCircuit(1)
        var_mdp_circ.ry(np.pi - self._prob_left_angle_param, 0)  # this directly encodes the value of the policy
        self.var_mdp_circ = var_mdp_circ

    def _init_var_a_phi_circ(self):
        self.var_a_phi_circ = self.var_mdp_circ

    def _init_var_grover_circ(self):
        qreg = QuantumRegister(self.var_a_phi_circ.num_qubits, name="g")
        var_grov_circ = QuantumCircuit(qreg, name="Grov")
        # z oracle
        var_grov_circ.x(qreg[-1])
        var_grov_circ.z(qreg[-1])
        var_grov_circ.x(qreg[-1])
        # A_phi^dagger
        a_phi_dg_circ = self.var_a_phi_circ.inverse()
        var_grov_circ = var_grov_circ.compose(a_phi_dg_circ, var_grov_circ.qubits)
        # S_0 gate
        var_grov_circ.z(qreg[-1])
        # A_phi
        var_grov_circ = var_grov_circ.compose(self.var_a_phi_circ, var_grov_circ.qubits)
        var_grov_circ = transpile(var_grov_circ, self.backend)
        self.var_grov_circ = var_grov_circ

    @staticmethod
    def phi_inv(x):
        return x

    def get_gt_return(self, prob_left):
        prob_right = 1 - prob_left
        win_prob_left = 1 - self.loose_prob_left
        win_prob_right = 1 - self.loose_prob_right
        # ground truth return
        gt_ret = prob_left * win_prob_left + prob_right * win_prob_right
        return gt_ret


class TwoStepBanditQPE(BanditQPE):
    """
    QPE for two steps of two-armed bandit as discussed in the corresponding sections.
    """

    def __init__(self, loose_prob_left, loose_prob_right, t_eval_qubits, backend, cache_dir, verbose):
        self._init_return_circ()
        super().__init__(loose_prob_left, loose_prob_right, t_eval_qubits, backend, cache_dir, verbose)

    def _init_return_circ(self):
        ret_circ = QuantumCircuit(6, name="G")
        ret_circ.cnot(1, 5)
        ret_circ.cnot(3, 5)
        ret_circ.toffoli(1, 3, 4)
        self.ret_circ = ret_circ

    def _init_var_mdp_circ(self):
        var_mdp_circ = QuantumCircuit(4, name="M")
        var_mdp_circ = var_mdp_circ.compose(self.var_pol_circ, [0])
        var_mdp_circ = var_mdp_circ.compose(self.env_circ, [0, 1])
        var_mdp_circ = var_mdp_circ.compose(self.var_pol_circ, [2])
        var_mdp_circ = var_mdp_circ.compose(self.env_circ, [2, 3])
        self.var_mdp_circ = var_mdp_circ

    def _init_var_a_phi_circ(self):
        # A gate
        var_a_circ = QuantumCircuit(6, name="A")
        var_a_circ = var_a_circ.compose(self.var_mdp_circ, [0, 1, 2, 3])
        var_a_circ = var_a_circ.compose(self.ret_circ, [0, 1, 2, 3, 4, 5])
        # A_phi gate
        var_a_phi_circ = QuantumCircuit(7, name="A_phi")
        var_a_phi_circ = var_a_phi_circ.compose(var_a_circ, [0, 1, 2, 3, 4, 5])
        var_a_phi_circ.cry(np.pi, 4, 6)
        var_a_phi_circ.cry(np.pi / 2, 5, 6)
        self.var_a_phi_circ = var_a_phi_circ

    @staticmethod
    def phi_inv(x):
        return 2 * x

    def get_gt_return(self, prob_left):
        # policy params
        prob_right = 1 - prob_left
        loose_prob_left = self.loose_prob_left
        win_prob_left = 1 - self.loose_prob_left
        loose_prob_right = self.loose_prob_right
        win_prob_right = 1 - self.loose_prob_right
        # ground truth return
        gt_ret = \
            2 * 1 * prob_left * win_prob_left * (prob_left * loose_prob_left + prob_right * loose_prob_right) + \
            2 * 1 * prob_right * win_prob_right * (prob_left * loose_prob_left + prob_right * loose_prob_right) + \
            1 * 2 * (prob_left * win_prob_left) ** 2 + \
            1 * 2 * (prob_right * win_prob_right) ** 2 + \
            2 * 2 * (prob_left * win_prob_left * prob_right * win_prob_right)
        return gt_ret


class OneStepBanditQPE(BanditQPE):
    """
    QPE for one step of two-armed bandit as discussed in the corresponding sections.
    """

    def __init__(self, loose_prob_left, loose_prob_right, t_eval_qubits, backend, cache_dir, verbose):
        self._init_return_circ()
        super().__init__(loose_prob_left, loose_prob_right, t_eval_qubits, backend, cache_dir, verbose)

    def _init_return_circ(self):
        ret_circ = QuantumCircuit(2, name="G")
        self.ret_circ = ret_circ

    def _init_var_mdp_circ(self):
        var_mdp_circ = QuantumCircuit(2, name="M")
        var_mdp_circ = var_mdp_circ.compose(self.var_pol_circ, [0])
        var_mdp_circ = var_mdp_circ.compose(self.env_circ, [0, 1])
        self.var_mdp_circ = var_mdp_circ

    def _init_var_a_phi_circ(self):
        self.var_a_phi_circ = self.var_mdp_circ

    @staticmethod
    def phi_inv(x):
        return x

    def get_gt_return(self, prob_left):
        prob_right = 1 - prob_left
        win_prob_left = 1 - self.loose_prob_left
        win_prob_right = 1 - self.loose_prob_right
        # ground truth return
        gt_ret = prob_left * win_prob_left + prob_right * win_prob_right
        return gt_ret


class BanditQPEExperiment:
    """
    Class to execute 'QPE experiments' for one or two steps of two armed bandit in parallel. 'QPE experiment' means that
    we store the output distribution of QPE along with some additional information to easily make plots etc.
    """

    def __init__(self, n_bandit_rounds, loose_prob_left, loose_prob_right, t_eval_qubits, backend,
                 cache_dir="./bandit_qpe_cache", verbose=False):
        self.n_bandit_rounds = n_bandit_rounds
        if n_bandit_rounds == 1:
            if loose_prob_left == 1 and loose_prob_right == 0:
                self.bandit_qpe_class = ReducedOneStepBanditQPE
            else:
                self.bandit_qpe_class = OneStepBanditQPE
        elif n_bandit_rounds == 2:
            self.bandit_qpe_class = TwoStepBanditQPE
        self.loose_prob_left = loose_prob_left
        self.loose_prob_right = loose_prob_right
        self.t_eval_qubits = t_eval_qubits
        self.backend = backend
        self.cache_dir = cache_dir
        self.verbose = verbose

        # if there is no cached QPE circuit, make one; this saves intialization time below
        cache_dir = f"{cache_dir}/{n_bandit_rounds}_rounds/"
        qpe_circ_id = f"{self.loose_prob_left}_{self.loose_prob_right}_{self.t_eval_qubits}_{self.backend}"
        if not os.path.exists(f"{cache_dir}/{qpe_circ_id}.pkl"):
            bandit_qpe = self.bandit_qpe_class(
                loose_prob_left=loose_prob_left,
                loose_prob_right=loose_prob_right,
                t_eval_qubits=t_eval_qubits,
                backend=backend,
                cache_dir=cache_dir,
                verbose=verbose,
            )
            # delete to save memory
            del bandit_qpe

    def print(self, txt):
        verbose_print(txt, self.verbose)

    def _init_qpe_worker(self):
        """
        Initialization function to make a BanditQPE object available to a multiprocessing.Pool worker
        """
        import multiprocessing
        worker_id = multiprocessing.current_process().ident
        self.print(f"WORKER {worker_id}: Loading QPE...")
        global bandit_qpe
        bandit_qpe = self.bandit_qpe_class(
            loose_prob_left=self.loose_prob_left,
            loose_prob_right=self.loose_prob_right,
            t_eval_qubits=self.t_eval_qubits,
            backend=self.backend,
            cache_dir=f"{self.cache_dir}/{self.n_bandit_rounds}_rounds/",
            verbose=False
        )
        self.print(f"WORKER {worker_id}: ...done with loading QPE!")

    def _run_qpe_worker(self, prob_left):
        """
        Let multiprocessing.Pool worker run QPE for policy with determined by prob_left
        :param prob_left: probability with which the left arm is chosen
        :return: dict with experimental quant_pol_iter_statistics
            {
                "prob_left": input prob_left
                "gt_ret": ground truth value of policy
                "qpe_ret_prob_dict": analytically calculated distribution of QPE outputs
            }
        """
        import multiprocessing
        worker_id = multiprocessing.current_process().ident
        self.print(f"WORKER {worker_id}: Start evaluating policy with prob_left={np.round(prob_left, 5)}...")
        start = datetime.now()
        # initialize output dict to which experimental quant_pol_iter_statistics are saved
        expermiental_results = {
            "prob_left": prob_left,
            "gt_ret": None,
            # "epsilon": None,
            "qpe_ret_prob_dict": None,
            # "qpe_oob_prob": None
        }
        gt_ret = bandit_qpe.get_gt_return(prob_left)
        expermiental_results["gt_ret"] = gt_ret
        # run qpe
        qpe_ret_prob_dict = bandit_qpe.get_return_dist_exact(prob_left)
        expermiental_results["qpe_ret_prob_dict"] = qpe_ret_prob_dict
        stop = datetime.now()
        self.print(
            f"WORKER {worker_id}: ...done with policy with prob_left={np.round(prob_left, 5)}! Took {stop - start}")
        return expermiental_results

    def run_parallel(self, prob_left_list, n_workers=1, output_file=None):
        """
        Run QPE experiments for all policies in prob_left list using a multiprocessing.Pool of n_workers workers
        :param prob_left_list: list of probabilities of choosing the left arm
        :param n_workers: number of multiprocessing.Pool workers
        :param output_file: give filepath if you want to save the quant_pol_iter_statistics to disk
        :return: list of quant_pol_iter_statistics of _run_qpe_worker
        """
        start = datetime.now()
        pool = Pool(n_workers, initializer=self._init_qpe_worker)
        qpe_experiment_results = pool.map(self._run_qpe_worker, prob_left_list)
        pool.close()
        pool.join()
        stop = datetime.now()
        self.print(f"Total time to evaluate {len(prob_left_list)} policies with {n_workers} workers: {stop - start}")
        if output_file is not None:
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(output_file, "wb") as f:
                pickle.dump(qpe_experiment_results, f)
        return qpe_experiment_results
