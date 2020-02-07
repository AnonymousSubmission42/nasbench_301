import os

import click
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import matplotlib.pyplot as plt
import numpy as np

from surrogate_models.utils import scatter_plot


def losses_over_time(runs, get_loss_from_run_fn=lambda r: r.loss, cmap=plt.get_cmap("tab10"), show=False):
    budgets = set([r.budget for r in runs])

    data = {}
    for b in budgets:
        data[b] = []

    for r in runs:
        if r.loss is None or r.loss == np.nan:
            continue
        b = r.budget
        t = r.time_stamps['finished']
        l = get_loss_from_run_fn(r)
        data[b].append((t, l))

    for b in budgets:
        data[b].sort()

    fig, ax = plt.subplots()

    for i, b in enumerate(budgets):
        data[b] = np.array(data[b])
        ax.scatter(data[b][:, 0], data[b][:, 1], color=cmap(i), label='b=%f' % b)

        ax.step(data[b][:, 0], np.minimum.accumulate(data[b][:, 1]), where='post')

    ax.set_title('Losses for different budgets over time')
    ax.set_xlabel('wall clock time [s]')
    ax.set_ylabel('loss')
    ax.legend()
    if show:
        plt.show()
    return (fig, ax)


@click.command()
@click.option("--run_name", help="Directory of Hpbandster run.", type=click.STRING)
def analysis(run_name):
    """
    Function to create plots of the current hpo runs.
    :param run_name:
    :return:
    """
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(run_name)

    # get all executed runs
    all_runs = result.get_all_runs()

    """
    # Plot mse loss vs 1 - test correlation
    losses = []
    corrs = []
    for conf in all_runs:
        if conf['info'] is not None and conf['loss'] is not None and conf['loss'] != np.nan and conf['loss'] < 1.0:
            loss = conf['loss']
            losses.append(loss)
            corrs.append(1 - conf['info'][0]['test_corr'])
    fig = scatter_plot(np.array(losses), np.array(corrs), 'MSE loss', '1 - Test Correlation', title=None)
    fig.savefig(os.path.join(run_name, 'correlation_mse_loss_vs_test_correlation.pdf'))
    plt.close()

    # Plot mse loss vs 1 - extrapolation correlation
    losses = []
    extra_corrs = []
    budgets = []
    for conf in all_runs:
        if conf['info'] is not None and conf['loss'] is not None and conf['loss'] != np.nan and conf['loss'] < 1.0:
            loss = conf['loss']
            losses.append(loss)
            extra_corr = 1 - conf['info'][0]['extrapolation_corr']
            extra_corrs.append(1 - conf['info'][0]['extrapolation_corr'])
            budgets.append(conf['budget'])

    

    # MSE loss vs. Num. Epochs
    plt.figure()
    plt.ylabel('1 - Extrapolation Correlation')
    plt.xlabel('MSE loss')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(extra_corrs), max(extra_corrs))
    ax.set_xscale('log')
    plt.xlim(min(losses), max(losses))
    plt.scatter(losses, extra_corrs, s=2, alpha=0.8, c=budgets)
    cbar = plt.colorbar()
    cbar.set_label('Num. Epochs', rotation=270)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, 'correlation_mse_loss_vs_extrapolation_correlation.pdf'))
    plt.close()

    # MSE loss vs. Num. Epochs
    plt.figure()
    plt.ylabel('1 - Extrapolation Correlation')
    plt.xlabel('Epochs')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.ylim(min(extra_corrs), max(extra_corrs))
    ax.set_xscale('log')
    plt.xlim(min(budgets), max(budgets))
    plt.scatter(budgets, extra_corrs, s=8, alpha=1.0)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, 'num_epochs_vs_extrapolation_correlation.pdf'))
    plt.close()

    print('Maximum of extrapolation correlation',
          1 - np.min(np.array(extra_corrs)[np.logical_not(np.isnan(extra_corrs))]))
    """
    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Here is how you get the incumbent (best configuration)
    inc_id = result.get_incumbent_id()
    if inc_id is not None:
        print('Incumbent ID', inc_id)
        # let's grab the run on the highest budgets
        inc_runs = result.get_runs_by_id(inc_id)
        inc_run = inc_runs[-1]

        # We have access to all information: the config, the loss observed during
        # optimization, and all the additional information
        inc_loss = inc_run.loss
        inc_config = id2conf[inc_id]['config']

        print(inc_run.info)
        # inc_val_corr = inc_run.info[0]['valid_corr']
        # inc_test_corr = inc_run.info[0]['test_corr']
        # extrapolation_corr = inc_run.info[0]['extrapolation_corr']

        print('Best found configuration:')
        print(inc_config)
        print(
            'It achieved validation MSE loss of %f (validation) and corr %f (validation)/ %f (test), extrapolation corr %f.' % (
                inc_loss, 1, 1, 1))#, inc_val_corr, inc_test_corr, extrapolation_corr))

    # Let's plot the observed losses grouped by budget,
    losses_over_time(all_runs)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, 'loss_over_time.pdf'))
    plt.close()

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, 'concurrent_runs_over_time.pdf'))
    plt.close()

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, 'finished_runs_over_time.pdf'))
    plt.close()

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)
    plt.tight_layout()
    plt.savefig(os.path.join(run_name, 'correlation_across_budgets.pdf'))
    plt.close()

    if "random" or "RS" not in run_name:
        # For model based optimizers, one might wonder how much the model actually helped.
        # The next plot compares the performance of configs picked by the model vs. random ones
        hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
        plt.tight_layout()
        plt.savefig(os.path.join(run_name, 'performance_histogram_model_vs_random.pdf'))
    plt.close()


if __name__ == '__main__':
    analysis()
