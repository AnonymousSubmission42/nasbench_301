import glob
import json
import os

import click
import seaborn as sns
from tqdm import tqdm

from surrogate_models.utils import ConfigLoader

sns.set_style('whitegrid')


@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
def nasbench_statistics(nasbench_data):
    config_loader = ConfigLoader('configspace.json')
    model_parameters = []

    train_errors = []
    test_errors = []
    val_errors = []
    num_layers_model = []
    init_channels_model = []
    num_epochs = []
    learning_rates = []
    run_times = []
    batch_sizes = []
    cutout_lengths = []
    weight_decays = []
    fidelity_statistics = {}
    for fidelity in range(7):
        results_paths = glob.glob(
            os.path.join(nasbench_data, 'run_*/results_fidelity_{}/results_*.json'.format(fidelity)))
        for config_path in tqdm(results_paths, desc='Reading dataset'):
            config_space_instance, val_accuracy, test_accuracy, json_file = config_loader[config_path]
            try:
                test_accuracy = json_file['test_accuracy']

                num_parameters = json_file['info'][0]['model_parameters']
                epochs = json_file['budget']
                train_accuracy = json_file['info'][0]['train_accuracy_final']
                val_accuracy = json_file['info'][0]['val_accuracy_final']

                num_layers = json_file['optimized_hyperparamater_config']['NetworkSelectorDatasetInfo:darts:layers']
                init_channels = json_file['optimized_hyperparamater_config'][
                    'NetworkSelectorDatasetInfo:darts:init_channels']
                run_time = json_file['runtime']
                learning_rate = json_file['optimized_hyperparamater_config']['OptimizerSelector:sgd:learning_rate']
                batch_size = json_file['optimized_hyperparamater_config']['CreateImageDataLoader:batch_size']
                cutout_length = json_file['optimized_hyperparamater_config']['ImageAugmentation:cutout_length']
                weight_decay = json_file['optimized_hyperparamater_config']['OptimizerSelector:sgd:weight_decay']

                weight_decays.append(weight_decay)
                cutout_lengths.append(cutout_length)
                run_times.append(run_time)
                model_parameters.append(num_parameters)
                num_epochs.append(epochs)
                batch_sizes.append(batch_size)

                train_errors.append(1 - train_accuracy / 100)
                test_errors.append(1 - test_accuracy / 100)
                val_errors.append(1 - val_accuracy / 100)

                num_layers_model.append(num_layers)
                init_channels_model.append(init_channels)
                learning_rates.append(learning_rate)

            except Exception as e:
                print('error', e, config_path, json_file)

        fidelity_statistics[fidelity] = {
            'min_validation_error': min(val_errors),
            'max_validation_error': max(val_errors),

            'min_test_error': min(test_errors),
            'max_test_error': max(test_errors),

            'min_runtime': min(run_times),
            'max_runtime': max(run_times)
        }

    json.dump(fidelity_statistics, open('surrogate_models/fidelity_statistics.json', 'w'))


if __name__ == "__main__":
    nasbench_statistics()
