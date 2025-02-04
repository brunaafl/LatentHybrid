import torch
from moabb.datasets import BNCI2014001, Cho2017, Lee2019_MI, Schirrmeister2017, PhysionetMI
from moabb.paradigms import MotorImagery, LeftRightImagery

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from moabb.utils import set_download_dir

from util import parse_args, set_determinism, set_run_dir

from hybrid_model import HybridModel
from hybrid_evaluation import HybridEvaluation
from hybrid_transform import HybridAggregateTransform
from hybrid_classifier import define_hybrid_clf

from paradigm import MotorImagery_

import torchinfo

import numpy as np

from time import time

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

    if args.dataset == 'BNCI2014001':
        dataset = BNCI2014001()
    elif args.dataset == 'Cho2017':
        dataset = Cho2017()
    elif args.dataset == 'Lee2019_MI':
        dataset = Lee2019_MI()
    elif args.dataset == 'Schirrmeister2017':
        dataset = Schirrmeister2017()
    elif args.dataset == 'PhysionetMI':
        dataset = PhysionetMI()
        paradigm = MotorImagery_(resample=250, metric='accuracy')


    paradigm = MotorImagery_(events=events, channels=channels, n_classes=len(events), metric='accuracy', resample=250)

    events = ["left_hand", "right_hand"]
    n_classes = len(events)

    for subj in dataset.subject_list:

        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[subj], return_epochs=False)
        print(X.shape)


    print("---------------------------------------")

    # return results


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
