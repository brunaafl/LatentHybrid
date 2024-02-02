"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>

Baseline script to analyse the EEG Dataset.

"""
import mne
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from glob import glob

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from braindecode.preprocessing import preprocess, Preprocessor

from alignment import euclidean_alignment
from dataset import read_dataset
from model_validation import split_dataset
from util import parse_args, set_determinism, set_run_dir
from train import train_all_loo
import torch
from braindecode.models import EEGNetv4

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
    Data_subjects = split_dataset(windows_dataset, subject_ids)

    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]
    n_classes = args.n_classes

    model = EEGNetv4(
        n_channels,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
        drop_prob=0.5
    )

    # Send model to GPU
    if cuda:
        model.cuda()

    aux_without_ea = []
    models_list, predicts_list= train_all_loo(model, Data_subjects, device, subject_ids)
    # save results
    for idx, test_subj in enumerate(predicts_list):
        if args.dataset_type == "alignment":
            filename_ = f"loo_part_1_with_EA_subject_{idx}.npz"
        else:
            filename_ = f"loo_part_1_without_EA_subject_{idx}.npz"
        np.savez(file=filename_, y_pred=test_subj[0], y_true=test_subj[1])
    # Calculate the accuracy score
    file_list_ = glob(f"*without_EA*.npz")
    file_list_.sort()
    without_ea = []

    for filename_ in file_list_:
        np_struct = np.load(filename_)
        ac = roc_auc_score(y_true=np_struct['y_true'], y_pred=np_struct['y_pred'])
        without_ea.append(ac)
    without_ea = np.array(without_ea)
    aux_without_ea.append(without_ea)

    table_exp = pd.DataFrame([aux_without_ea[i, :] for i in range(aux_without_ea.shape[0])]).T
    table_exp.columns = [f'exp_{i}' for i in range(aux_without_ea.shape[0])]
    print(table_exp)

    print("---------------------------------------")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
