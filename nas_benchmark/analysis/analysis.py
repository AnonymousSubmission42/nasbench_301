import glob
import json
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import interp1d
from tqdm import tqdm

sns.set_style('whitegrid')


def get_available_results_for_fidelity(results, fidelity, algos):
    optimizer_results = 'nas_benchmark/results'
    optimizer_results_dict = {}
    for algo in algos:
        algo_results_path = os.path.join(optimizer_results, algo)
        algo_results_for_fidelity = glob.glob(os.path.join(algo_results_path, '*{}*'.format(fidelity)))
        if len(algo_results_for_fidelity) > 0:
            assert len(algo_results_for_fidelity) == 1, 'Multiple fidelities were matched.'
            results = np.array(json.load(open(algo_results_for_fidelity[0])))
            if algo == 'smac':
                results = np.array([dict['validation_error'] for dict in results])

            optimizer_results_dict[algo] = (np.mean(results, axis=0), np.std(results, axis=0))
    return optimizer_results_dict


@click.command()
@click.option('--result_file_dir', type=click.STRING, help='path to results_file', required=True)
def plot_results(result_file_dir):
    # Preprocess the data
    fidelity_algo_dict = {str(fidelity): {} for fidelity in range(7)}
    log_path = os.path.join('nas_benchmark', result_file_dir, 'plot_results')
    os.makedirs(log_path)
    result_files = glob.glob(os.path.join(result_file_dir, '*.json'))
    for result_file in result_files:
        results = json.load(open(result_file, 'r'))
        optimizer_name = results['algo']
        optimizer_results = results['results']

        for fidelity, value in optimizer_results.items():
            fidelity_algo_dict[fidelity][optimizer_name] = value

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for fidelity, algos in tqdm(fidelity_algo_dict.items()):
        plt.figure(figsize=(4, 3))
        plt.title('Fidelity {}'.format(fidelity))
        for i, (algo_name, algo_data) in enumerate(algos.items()):
            color = colors[i]
            label = algo_name
            for data in algo_data:
                incumbent_interpolation = interp1d(data['runtime'], data['validation_error'], kind='previous')
                new_x = np.linspace(min(data['runtime']) + 1, max(data['runtime']), 10000, dtype=np.int)
                plt.plot(new_x, incumbent_interpolation(new_x), label=label, alpha=0.5, c=color)
                label = None
        ax = plt.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlabel('Runtime (s)')
        plt.ylabel('Validation error')
        plt.legend()
        plt.grid(True, which='both', ls='-', alpha=0.1)
        plt.tight_layout()

        plt.savefig(os.path.join(log_path, 'discrete_optimizer_analysis_fidelity_{}.png'.format(fidelity)),
                    dpi=300)
        plt.close()


if __name__ == "__main__":
    plot_results()
