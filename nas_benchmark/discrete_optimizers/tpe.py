import argparse
import json
import os
from copy import deepcopy

import ConfigSpace
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from nas_benchmark.benchmarks.nasbench_301 import NASBench301_arch_only

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--output_path', default="nas_benchmark/results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
parser.add_argument('--data_dir', default="./", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--surrogate_model_dir', type=str, nargs='?', help='path to surrogate model')
parser.add_argument('--n_repetitions', default=30, type=int, help='number of repetitions')

args = parser.parse_args()

for fidelity in range(7):
    b = NASBench301_arch_only(surrogate_model_dir=args.surrogate_model_dir, fidelity=fidelity)

    output_path = os.path.join(args.output_path, "tpe")
    os.makedirs(os.path.join(output_path), exist_ok=True)

    cs = b.get_configuration_space()

    space = {}
    for h in cs.get_hyperparameters():
        if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:
            space[h.name] = hp.quniform(h.name, 0, len(h.sequence) - 1, q=1)
        elif type(h) == ConfigSpace.hyperparameters.CategoricalHyperparameter:
            space[h.name] = hp.choice(h.name, h.choices)
        elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:
            space[h.name] = hp.quniform(h.name, h.lower, h.upper, q=1)
        elif type(h) == ConfigSpace.hyperparameters.UniformFloatHyperparameter:
            space[h.name] = hp.uniform(h.name, h.lower, h.upper)


    def objective(x):
        config = deepcopy(x)
        for h in cs.get_hyperparameters():
            if type(h) == ConfigSpace.hyperparameters.OrdinalHyperparameter:

                config[h.name] = h.sequence[int(x[h.name])]

            elif type(h) == ConfigSpace.hyperparameters.UniformIntegerHyperparameter:

                config[h.name] = int(x[h.name])
        y = 1 - b.objective_function(config) / 100

        return {
            'config': config,
            'loss': y,
            'status': STATUS_OK}


    trials = Trials()
    best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=args.n_iters,
                trials=trials)

    res = b.get_results()

    fh = open(os.path.join(output_path, 'run_%d.json' % args.run_id), 'w')
    json.dump(res, fh)
    fh.close()
