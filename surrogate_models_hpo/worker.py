import json
import os
import random
import time

import matplotlib
import numpy as np
from hpbandster.core.worker import Worker

from surrogate_models import utils

matplotlib.use('Agg')


class HPOWorker(Worker):
    def __init__(self, eta, min_budget, max_budget, model, data_root, data_config_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.model = model
        self.data_root = data_root
        self.data_config_path = data_config_path

    def compute(self, config, budget, config_id, working_directory):
        model_config = config
        model_config['model'] = self.model

        dest_dir = os.path.join(working_directory, "_".join(map(str, config_id)))

        # Seed
        seed = random.randint(0, 10000)

        # Create log directory
        log_dir = os.path.join(dest_dir, self.model + '/{}-{}'.format(time.strftime("%Y%m%d-%H%M%S"), seed))
        os.makedirs(log_dir)

        # Load config
        data_config = json.load(open(self.data_config_path, 'r'))

        if 'gnn' in self.model:
            model_config['epochs'] = int(budget)
        elif 'lgb' in self.model or 'xgb' in self.model:
            model_config['param:num_rounds'] = int(budget)
        elif 'random_forrest' in self.model or 'sklearn_forest' in self.model or 'svr' in self.model:
            print("Budget not considered for random forrests and svrs.")
        else:
            raise NotImplementedError('For other methods the budget is not yet defined.')

        # Instantiate surrogate model
        surrogate_model = utils.model_dict[self.model](data_root=self.data_root, log_dir=log_dir, seed=seed,
                                                       model_config=model_config, data_config=data_config)

        # Train and validate the model on the available data
        train_result = surrogate_model.train()

        # Test the model
        test_result = surrogate_model.test()

        # Convert numpy floats to python floats because of the serialization of BOHB.
        convert_to_python_primitives = lambda input_dict: {k: getattr(v, "tolist", lambda: v)() for k, v in
                                                           input_dict.items()}
        # Create dict keys
        train_result = convert_to_python_primitives(self.change_key_prefix(train_result, "val"))
        test_result = convert_to_python_primitives(self.change_key_prefix(test_result, "test"))

        val_obj = train_result["val_mse"]

        # Validate the extrapolation
        # extrapolation_corr = extrapolation(log_dir, nasbench_data=self.data_root, surrogate_model=surrogate_model)

        ret_dict = {'loss': val_obj,
                    'info': [{**train_result, **test_result}]}
        return ret_dict

    def change_key_prefix(self, d, prefix):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, np.generic):
                new_dict[prefix + "_" + key] = float(value.item())
            else:
                new_dict[prefix + "_" + key] = float(value)
        return new_dict

    def get_config_space(self):
        return utils.get_model_configspace(model=self.model)


if __name__ == "__main__":
    worker = HPOWorker(eta=3, min_budget=5, max_budget=5, run_id=1, data_root='/home/anonymous/NasBench301_v0.4',
                       data_config_path='surrogate_models/configs/data_configs/diagonal_plus_off_diagonal.json',
                       model='gnn_vs_gae')
    cs = worker.get_config_space()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, config_id=(3, 1), working_directory='.')
    print(res)
