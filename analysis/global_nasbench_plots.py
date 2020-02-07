from datetime import datetime
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from surrogate_models.utils import ConfigLoader

sns.set_style('whitegrid')


@click.command()
@click.option('--nasbench_data', type=click.STRING, help='path to nasbench root directory')
def plotting(nasbench_data):
    current_time = datetime.now()
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
    results_paths = [filename for filename in Path(nasbench_data).rglob('*.json')]  # + \
    # glob.glob(os.path.join(nasbench_data, 'random_fidelities/results_fidelity_*/results_*.json'))

    for config_path in tqdm(results_paths, desc='Reading dataset'):
        if "surrogate_model" in str(config_path):
            continue
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

    # Compute general statistic
    result_statistics = {
        'validation_error': [np.mean(val_errors), np.std(val_errors)],
        'test_error': [np.mean(test_errors), np.std(test_errors)],
        'learning_rate': [np.mean(learning_rates), np.std(learning_rates)],
        'num_layers': [np.mean(num_layers_model), np.std(num_layers_model)],
        'epochs': [np.mean(num_epochs), np.std(num_epochs)],
        'init_channels': [np.mean(init_channels_model), np.std(init_channels_model)],
        'weight_decay': [np.mean(weight_decays), np.std(weight_decays)],
    }
    # json.dump(result_statistics, open('surrogate_models/dataset_statistic.json', 'w'))

    print('corr val test (kendall):', kendalltau(val_errors, test_errors))
    print('corr train val (kendall):', kendalltau(train_errors, val_errors))

    print('corr val test (spearman):', spearmanr(val_errors, test_errors))
    print('corr train val (spearman):', spearmanr(train_errors, val_errors))

    print('total runtime:', np.sum(run_times))
    print('best performance (test accuracy):', 100-np.min(test_errors)*100)
    print('dataset statistics', result_statistics)

    # WEIGHT DECAY VS. VAL ERROR
    decoupled_weight_decay = np.array(weight_decays) / np.array(learning_rates)
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Weight Decay / Learning Rate')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(val_errors), max(val_errors))
    ax.set_xscale('log')
    plt.xlim(min(decoupled_weight_decay), max(decoupled_weight_decay))
    plt.scatter(decoupled_weight_decay, val_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/decoupled_weight_decay_vs_val_error.png', dpi=600)
    plt.close()

    # PARAMETERS VS. VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(val_errors), max(val_errors))
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, val_errors, s=2, alpha=0.15, c=run_times, norm=matplotlib.colors.LogNorm(),
                cmap=plt.get_cmap('magma_r'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Runtime (s)', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_val_error.png', dpi=600)
    plt.close()

    # CUTOUT LENGTH VS. VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Cutout Length')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(val_errors), max(val_errors))
    ax.set_xscale('log')
    plt.xlim(min(cutout_lengths), max(cutout_lengths))
    plt.scatter(cutout_lengths, val_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/cutout_length_vs_val_error.png', dpi=600)
    plt.close()

    # BATCH SIZE VS. VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Batch Size')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(val_errors), max(val_errors))
    ax.set_xscale('log')
    plt.xlim(min(batch_sizes), max(batch_sizes))
    plt.scatter(batch_sizes, val_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/batch_size_vs_val_error.png', dpi=600)
    plt.close()

    # PARAMETERS VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Test error')
    plt.xlabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, test_errors, s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Validation error', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_test_error.png', dpi=600)
    plt.close()

    # TRAIN ERROR VS. VAL ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Validation error')
    plt.xlabel('Train error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(val_errors), max(val_errors))
    ax.set_xscale('log')
    plt.xlim(min(train_errors), max(train_errors))
    plt.scatter(train_errors, val_errors, s=2, alpha=0.15, c=test_errors, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Test error', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/train_error_vs_val_error.png', dpi=600)
    plt.close()

    # PARAMETERS VS. RUNTIME
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Runtime (s)')
    plt.xlabel('Number Model Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(run_times), max(run_times))
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, run_times, s=1, alpha=0.2, c=val_errors, norm=matplotlib.colors.LogNorm(), cmap=plt.get_cmap('magma'))
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Validation Error', rotation=270, labelpad=15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_runtimes.png', dpi=600, bbox_inches='tight')
    plt.close()

    # PARAMETERS VS. RUNTIME
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Runtime')
    plt.xlabel('Num Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(run_times), max(run_times))
    ax.set_xscale('log')
    plt.xlim(min(model_parameters), max(model_parameters))
    plt.scatter(model_parameters, run_times, s=2, alpha=0.15, c=init_channels_model, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Initial Channels', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_runtimes_vs_init_channels.png', dpi=600)
    plt.close()

    # LEARNING RATE VS. EPOCHS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Epochs')
    plt.xlabel('Learning Rate')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(num_epochs), max(num_epochs))
    ax.set_xscale('log')
    plt.xlim(min(learning_rates), max(learning_rates))
    plt.scatter(learning_rates, num_epochs, s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Validation Error', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/learning_rate_vs_epochs_vs_validation_error.png', dpi=600)
    plt.close()

    # WEIGHT DECAY VS. EPOCHS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.ylabel('Epochs')
    plt.xlabel('Weight Decay')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(num_epochs), max(num_epochs))
    ax.set_xscale('log')
    plt.xlim(min(weight_decays), max(weight_decays))
    plt.scatter(weight_decays, num_epochs, s=2, alpha=0.15, c=val_errors, norm=matplotlib.colors.LogNorm())
    cbar = plt.colorbar()
    cbar.set_label('Validation Error', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/weight_decay_vs_epochs_vs_validation_error.png', dpi=600)
    plt.close()

    # PARAMETERS VS. INIT-CHANNELS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Init Channels')
    plt.ylabel('Num. Parameters')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlim(min(init_channels_model), max(init_channels_model))
    # ax.set_xscale('log')
    plt.ylim(min(model_parameters), max(model_parameters))
    plt.scatter(init_channels_model, model_parameters, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/parameters_vs_init_channels.png', dpi=600)
    plt.close()

    # LEARNING RATE VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Learning rate')
    plt.ylabel('Test error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    plt.xlim(min(learning_rates), max(learning_rates))
    plt.scatter(learning_rates, test_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/lr_vs_test_error.png', dpi=600)
    plt.close()

    # LAYERS VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Num. Layers')
    plt.ylabel('Test error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    ax.set_xscale('log')
    plt.xlim(min(num_layers_model), max(num_layers_model))
    plt.scatter(num_layers_model, test_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/num_layers_vs_test_error.png', dpi=600)
    plt.close()

    # INIT CHANNELS VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Initial channels')
    plt.ylabel('Test error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    ax.set_xscale('log')
    plt.xlim(min(init_channels_model), max(init_channels_model))
    plt.scatter(init_channels_model, test_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/init_channels_vs_test_error.png', dpi=600)
    plt.close()

    # EPOCH VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Training Epochs')
    plt.ylabel('Test error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    ax.set_xscale('log')
    plt.xlim(min(num_epochs), max(num_epochs))
    plt.scatter(num_epochs, test_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/num_epochs_vs_test_error.png', dpi=600)
    plt.close()

    # INIT CHANNELS VS. NUM LAYERS
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Initial channels')
    plt.ylabel('Number layers')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(num_layers_model), max(num_layers_model))
    ax.set_xscale('log')
    plt.xlim(min(init_channels_model), max(init_channels_model))
    plt.scatter(init_channels_model, num_layers_model, s=2, alpha=0.15, c=test_errors)
    plt.colorbar()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/init_channels_vs_num_layers.png', dpi=600)
    plt.close()

    # VAL ERROR VS. TEST ERROR
    plt.figure(figsize=(4, 3))
    # plt.title('NAS-Bench-301 {}, num. models: {}'.format(current_time.strftime("%b %d %Y %H:%M"), len(test_errors)))
    plt.xlabel('Validation error')
    plt.ylabel('Test error')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(test_errors), max(test_errors))
    ax.set_xscale('log')
    plt.xlim(min(val_errors), max(val_errors))
    plt.scatter(val_errors, test_errors, s=2, alpha=0.15)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig('analysis/plot_export/val_error_vs_test_error.png', dpi=600)
    plt.close()


if __name__ == "__main__":
    plotting()
