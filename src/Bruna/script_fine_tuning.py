"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>

Baseline script to analyse the EEG Dataset.

"""
import copy

import mne
import numpy as np
import pandas as pd
from braindecode.datasets import BaseConcatDataset

from omegaconf import OmegaConf
from glob import glob

from sklearn.metrics import balanced_accuracy_score

from braindecode.preprocessing import preprocess, Preprocessor

from alignment import euclidean_alignment
from dataset import read_dataset, ft_test_data, create_ft_dataset
from model_validation import split_dataset
from util import parse_args, set_determinism, set_run_dir
from train import train_all_loo, fine_tuning, init_model
import torch
from braindecode.models import EEGNetv4
from model_validation import split_train_val

mne.set_log_level("ERROR")

"""
For the joint model
"""


def main(args):
    """

    Parameters
    ----------
    args : object
    """

    # OmegaConf = lib to create objects from something
    config = OmegaConf.load(args.config_file)
    # Setting run information
    set_determinism(seed=config.seed)

    # check if GPU is available, if True chooses to use it
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'

    # Set directory: returns path and name of the experiment
    run_dir, experiment_name = set_run_dir(config, args)

    # Loading data
    windows_dataset = read_dataset(
        config, dataset_name=args.dataset_name, subject_ids=None
    )

    # if we want alignment
    if args.dataset_type == "alignment":
        aligner = [Preprocessor(euclidean_alignment, apply_on_array=True)]
        windows_dataset = preprocess(windows_dataset, aligner)

    # Split dataset in subjects
    subject_ids = args.subject_ids
    run = args.run
    Data_subjects = split_dataset(windows_dataset, subject_ids)

    bac_subj = []
    for subj in subject_ids:
        train_idx = copy.deepcopy(subject_ids)
        train_idx.remove(subj)
        Train = BaseConcatDataset([Data_subjects[f'{i}'] for i in train_idx])
        Test_runs, Test_T, Test_E = ft_test_data(subj, subject_ids, run, Data_subjects)
        Train_test = create_ft_dataset(Train, Test_runs)
        Test = Test_E

        # Create model
        n_channels = windows_dataset[0][0].shape[0]
        input_window_samples = windows_dataset[0][0].shape[1]
        n_classes = args.n_classes
        model = init_model(n_channels, n_classes, input_window_samples)
        if cuda:
            model.cuda()

        # Salvar os dados
        bac_runs = fine_tuning(model, device, subj, Test, Train_test)
        bac_subj.append(bac_runs)

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
