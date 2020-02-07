import numpy as np
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.tae.execute_func import ExecuteTAFuncDict

from nas_benchmark.benchmarks.nasbench_301 import NASBench301_arch_only


def smac_frontend(surrogate_model_dir, runtime_model_dir, n_repetitions, n_iters, n_trees=10, random_fraction=0.33,
                  max_feval=4):
    results_per_fidelity = {}
    for fidelity in range(7):
        results_per_random_seed = []
        for seed in range(n_repetitions):
            print("##### Seed {} #####".format(seed))
            np.random.seed(seed)
            b = NASBench301_arch_only(surrogate_model_dir, runtime_model_dir, fidelity=fidelity)

            cs = b.config_space

            scenario = Scenario({"run_obj": "quality",
                                 "runcount-limit": n_iters,
                                 "cs": cs,
                                 "deterministic": "false",
                                 "initial_incumbent": "RANDOM",
                                 "output_dir": ""})

            def objective_function(config, **kwargs):
                y = b.objective_function(config.get_dictionary())
                return float(y)

            tae = ExecuteTAFuncDict(objective_function, use_pynisher=False)
            smac = SMAC(scenario=scenario, tae_runner=tae)

            # probability for random configurations
            smac.solver.random_configuration_chooser.prob = random_fraction
            smac.solver.model.rf_opts.num_trees = n_trees
            # only 1 configuration per SMBO iteration
            smac.solver.scenario.intensification_percentage = 1e-10
            smac.solver.intensifier.min_chall = 1
            # maximum number of function evaluations per configuration
            smac.solver.intensifier.maxR = max_feval

            smac.optimize()

            res = b.get_results()
            results_per_random_seed.append(res)
        results_per_fidelity[fidelity] = results_per_random_seed
    return results_per_fidelity
