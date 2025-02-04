"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
Baseline script to analyse the EEG Dataset.
"""

import torch
import numpy as np
import pandas as pd

from moabb.datasets import BNCI2014001, Cho2017, Lee2019_MI, Schirrmeister2017, PhysionetMI

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from moabb.utils import set_download_dir

from mne.decoding import Scaler

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
from train import define_clf, init_model
from evaluation import shared_model, online_shared
from util import parse_args, set_determinism, set_run_dir
from paradigm import MotorImagery_

"""
For the shared model
"""


def main(args):
    """
    Parameters
    ----------
    args : object
    """
    config = OmegaConf.load(args.config_file)
    # Setting run information
    set_determinism(seed=config.seed)
    # Set download dir
    run_dir, experiment_name = set_run_dir(config, args)
    set_download_dir(config.dataset.path)
    cuda = (
        torch.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    # Define paradigm and datasets
    events = ["right_hand", "left_hand"]
    channels = ["Fz", "FC3", "FCz", "FC4", "C5", "FC1", "FC2",
                "C3", "C4", "Cz", "C6", "CPz", "C1", "C2",
                "CP2", "CP1", "CP4", "CP3", "Pz", "P2", "P1", "POz"]

    paradigm = MotorImagery_(events=events, channels=channels, n_classes=len(events), metric='accuracy', resample=250)

    if args.dataset == 'BNCI2014001':
        dataset = BNCI2014001()
    elif args.dataset == 'Cho2017':
        dataset = Cho2017()
    elif args.dataset == 'Lee2019_MI':
        dataset = Lee2019_MI()
    elif args.dataset == 'Schirrmeister2017':
        if args.ea == 'rest-alignment':
            events = ["right_hand", "left_hand", "rest"]
        dataset = Schirrmeister2017()
    elif args.dataset == 'PhysionetMI':
        dataset = PhysionetMI()
        paradigm = MotorImagery_(resample=250, metric='accuracy')

    events = ["left_hand", "right_hand"]
    n_classes = len(events)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1], return_epochs=True)
    n_chans = X.get_data().shape[1]
    input_window_samples = X.get_data().shape[2]
    ea = config.ea.batch

    model = init_model(n_chans, n_classes, input_window_samples, config=config)
    # Send model to GPU
    if cuda:
        model.cuda()

    # Create Classifier
    clf = define_clf(model, config)

    create_dataset_with_align = TransformaParaWindowsDatasetEA(ea)
    create_dataset_with_ralign = TransformaParaWindowsDatasetEA(ea, atype='riemann')
    create_dataset = TransformaParaWindowsDataset()

    pipes = {}

    pipe_with_align = Pipeline([("Braindecode_dataset", create_dataset_with_align),
                                ("Net", clone(clf))])
    pipe_with_ralign = Pipeline([("Braindecode_dataset", create_dataset_with_ralign),
                                 ("Net", clone(clf))])
    pipe = Pipeline([("Braindecode_dataset", create_dataset),
                     ("Net", clone(clf))])

    if args.ea == 'alignment':
        pipes[f"{config.model.type}_EA"] = pipe_with_align
    elif args.ea == 'r-alignment':
        pipes[f"{config.model.type}_RA"] = pipe_with_ralign
    else:
        pipes[f"{config.model.type}_Without_EA"] = pipe

    # Define evaluation and train
    # First, offline, zero-shot and online with 1 run for EA
    results = shared_model(dataset, paradigm, pipes, run_dir, config, align=args.ea)

    print(results.head())

    # Save results
    results.to_csv(f"{run_dir}/{experiment_name}_results.csv")

    print("---------------------------------------")

    # return results


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
