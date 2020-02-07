import logging

import hpbandster.core.nameserver as hpns
import numpy as np
from hpbandster.core.worker import Worker
from hpbandster.optimizers.bohb import BOHB

from nas_benchmark.benchmarks.nasbench_301 import NASBench301_arch_only, NASBench301_arch_hyp

logging.basicConfig(level=logging.ERROR)


def bohb_frontend(surrogate_model_dir, runtime_model_dir, n_repetitions, n_iters, benchmark='NASBench301_fixed_hps',
                  min_bandwidth=0.3, num_samples=64, random_fraction=.33, bandwidth_factor=3):
    results_per_fidelity = {}
    results_per_random_seed = []
    for seed in range(n_repetitions):
        print("##### Seed {} #####".format(seed))
        np.random.seed(seed)
        if benchmark == 'NASBench301_fixed_hps':
            b = NASBench301_arch_only(surrogate_model_dir=surrogate_model_dir, runtime_model_dir=runtime_model_dir,
                                      fidelity=6)
            min_budget = 1
            max_budget = 5
        elif benchmark == 'NASBench301':
            b = NASBench301_arch_hyp(surrogate_model_dir=surrogate_model_dir, runtime_model_dir=runtime_model_dir,
                                     fidelity=6)
            min_budget = 1
            max_budget = 5

        cs = b.config_space

        class MyWorker(Worker):
            def compute(self, config, budget, **kwargs):
                y = b.objective_function(config, eval_fidelity=budget)  # int(np.round(budget)))
                return ({'loss': float(y)})

        hb_run_id = '0'

        NS = hpns.NameServer(run_id=hb_run_id, host='localhost', port=0)
        ns_host, ns_port = NS.start()

        num_workers = 1

        workers = []
        for i in range(num_workers):
            w = MyWorker(nameserver=ns_host, nameserver_port=ns_port, run_id=hb_run_id, id=i)
            w.run(background=True)
            workers.append(w)

        bohb = BOHB(configspace=cs, run_id=hb_run_id, eta=2, min_budget=min_budget, max_budget=max_budget,
                    nameserver=ns_host, nameserver_port=ns_port, num_samples=num_samples,
                    random_fraction=random_fraction, bandwidth_factor=bandwidth_factor, ping_interval=10,
                    min_bandwidth=min_bandwidth)

        results = bohb.run(n_iters, min_n_workers=num_workers)

        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

        res = b.get_results()
        results_per_random_seed.append(res)
    results_per_fidelity[max_budget] = results_per_random_seed
    return results_per_fidelity
