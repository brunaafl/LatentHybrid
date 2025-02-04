import pandas as pd

import moabb.analysis.plotting as moabb_plt
from moabb.analysis.meta_analysis import (  # noqa: E501
    compute_dataset_statistics,
    find_significant_differences,
)
import matplotlib.pyplot as plt

from util import set_run_dir
from omegaconf import OmegaConf


def statistical_analysis(args, results1_csv, results2_csv):
    """
    Execute statistical analysis and plot graphs

    :param results2_csv:
    :param results1_csv:
    :param args:
    :return:
    """

    results1 = pd.read_csv(results1_csv)
    results2 = pd.read_csv(results2_csv)

    # Set run dir
    config = OmegaConf.load(args.config_file)
    run_dir, experiment_name = set_run_dir(config, args)

    # Concat results from different pipelines
    results = pd.concat([results1, results2])

    fig, color_dict = moabb_plt.score_plot(results)
    fig.savefig(f"{run_dir}/score_plot_models.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig = moabb_plt.paired_plot(results, "EEGNetv4_EA", "EEGNetv4_Without_EA")
    fig.savefig(f"{run_dir}/paired_score_plot_models.pdf", format='pdf', dpi=300, bbox_inches='tight')

    stats = compute_dataset_statistics(results)
    P, T = find_significant_differences(stats)

    fig = moabb_plt.meta_analysis_plot(stats, "EEGNetv4_EA", "EEGNetv4_Without_EA")
    fig.savefig(f"{run_dir}/meta_analysis_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()

    fig = moabb_plt.summary_plot(P, T)
    fig.savefig(f"{run_dir}/meta_analysis_summary_plot.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
