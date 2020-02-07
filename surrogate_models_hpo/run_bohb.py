import argparse
import logging
import pickle
import time

import hpbandster.core.result as hputil
from ConfigSpace.read_and_write import json as config_space_json_r_w
from hpbandster.optimizers.bohb import BOHB
from hpbandster.utils import *

from surrogate_models import utils
from surrogate_models_hpo import hpo_utils
from surrogate_models_hpo.worker import HPOWorker as worker

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description='Run BOHB on CIFAR10 search space.')
parser.add_argument('--num_iterations', type=int, help='number of Hyperband iterations performed.', default=64)
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='eth1')
parser.add_argument('--working_directory', type=str, help='directory where to store the live rundata', default=None)
parser.add_argument('--array_id', type=int, default=1)
parser.add_argument('--total_num_workers', type=int, default=20)
parser.add_argument('--min_budget', type=int, default=4, help='minimum budget given to BOHB (in epochs).')
parser.add_argument('--max_budget', type=int, default=32, help='maximum budget given to BOHB (in epochs).')
parser.add_argument('--eta', type=int, default=2, help='Multiplicative factor across budgets.')
parser.add_argument('--model', choices=utils.model_dict.keys(), help='Which model to use.')
parser.add_argument('--data_config_path', type=str, help='Path to config.json',
                    default='surrogate_models/configs/data_configs/diagonal_plus_off_diagonal.json')
parser.add_argument('--data_root', type=str, help='Root directory of the nasbench data.')

args = parser.parse_args()

min_budget = args.min_budget
max_budget = args.max_budget
eta = args.eta

if args.array_id == 1:
    os.makedirs(args.working_directory, exist_ok=True)

    NS = hpo_utils.NameServer(run_id=args.run_id, nic_name="eth0", working_directory=args.working_directory)
    ns_host, ns_port = NS.start()

    # BOHB is usually so cheap, that we can afford to run a worker on the master node, too.
    worker = worker(min_budget=min_budget, max_budget=max_budget, eta=eta,
                    nameserver=ns_host, nameserver_port=ns_port, run_id=args.run_id,
                    model=args.model, data_config_path=args.data_config_path, data_root=args.data_root)
    worker.run(background=True)

    # Dump the configspace to the directory
    config_space = worker.get_config_space()
    with open(os.path.join(args.working_directory, 'configspace.json'), 'w') as f:
        f.write(config_space_json_r_w.write(config_space))

    # instantiate BOHB and run it
    result_logger = hputil.json_result_logger(directory=args.working_directory, overwrite=True)

    HPB = BOHB(configspace=worker.get_config_space(),
               working_directory=args.working_directory,
               run_id=args.run_id,
               eta=eta, min_budget=min_budget, max_budget=max_budget,
               host=ns_host,
               nameserver=ns_host,
               nameserver_port=ns_port,
               ping_interval=3600,
               result_logger=result_logger
               )

    res = HPB.run(n_iterations=args.num_iterations,
                  min_n_workers=args.total_num_workers
                  # BOHB can wait until a minimum number of workers is online before starting
                  )

    with open(os.path.join(args.working_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    HPB.shutdown(shutdown_workers=True)
    NS.shutdown()

else:
    time.sleep(30)

    host = hpo_utils.nic_name_to_host("eth0")

    worker = worker(min_budget=min_budget, max_budget=max_budget, eta=eta, host=host, run_id=args.run_id,
                    model=args.model, data_config_path=args.data_config_path, data_root=args.data_root)
    worker.load_nameserver_credentials(args.working_directory)
    worker.run(background=False)
    exit(0)
