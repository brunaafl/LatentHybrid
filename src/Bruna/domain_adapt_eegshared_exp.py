"""
Authors: Bruno Aristimunha <b.aristimunha@gmail.com>
Baseline script to analyse the EEG Dataset.
"""
import warnings
from time import time

import braindecode
import moabb
import torchinfo
import torch

import numpy as np

from moabb.datasets import BNCI2014_001, BNCI2015_001, Lee2019_MI, Schirrmeister2017, PhysionetMI, Weibo2014
from moabb.paradigms import MotorImagery, LeftRightImagery

from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from moabb.utils import set_download_dir

from pipeline import TransformaParaWindowsDataset, TransformaParaWindowsDatasetEA
from shared_evaluation import EEGSharedEvaluation
from paradigm import MotorImagery_
from train import define_clf, init_model
from util import parse_args, set_determinism, set_run_dir

"""
For the joint model
"""

moabb.set_log_level("info")
warnings.filterwarnings("ignore")


def main(args):
    torch.set_num_threads(1)

    init_time = time()

    config = OmegaConf.load(args.config_file)
    eval_config = OmegaConf.load(args.eval_config_file)
    # Setting run information
    set_determinism(seed=config.seed)
    # Set download dir
    run_dir, experiment_name = set_run_dir(config, args)
    set_download_dir(config.dataset.path)
    cuda = (
        torch.cuda.is_available()
    )  # check if GPU is available, if True chooses to use it
    # Define paradigm and datasets

    print(braindecode.__version__)

    print(f"(1) Initial {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    events = ["right_hand", "left_hand"]

    if args.dataset == 'BNCI2014001':
        dataset = BNCI2014_001()
        ch=None
        subjects = dataset.subject_list
    elif args.dataset == 'Weibo2014':
        dataset = Weibo2014()
        ch = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3",
              "CP1", "CPz", "CP6", "CP4", "CP2"]
    elif args.dataset == 'BNCI2015001':
        dataset = BNCI2015_001()
        ch = None
        events = ["right_hand", "feet"]
    elif args.dataset == 'Schirrmeister2017':
        ch = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3",
              "CP1", "CPz", "CP6", "CP4", "CP2"]
        dataset = Schirrmeister2017()
        subjects = dataset.subject_list
        subjects.pop(0)
        dataset.subject_list = subjects
    elif args.dataset == 'PhysionetMI':
        dataset = PhysionetMI()
        paradigm = LeftRightImagery(resample=100.0)

    paradigm = MotorImagery_(events=events, n_classes=len(events), channels=ch, resample=250)

    datasets = [dataset]
    n_classes = len(events)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[2], return_epochs=True)
    sfreq = X.info['sfreq']
    X = X.get_data()
    n_chans = X.shape[1]
    input_window_samples = X.shape[2]
    len_run = config.train.len_run if args.ea else None

    model = init_model(n_chans, n_classes, input_window_samples, config=config)

    # Send model to GPU
    if cuda:
        model.cuda()

    # Create Classifier
    clf = define_clf(model, config, warm_start=True, experiment_name='EEGClassifier')

    create_dataset_with_align = TransformaParaWindowsDatasetEA(len_run, sfreq)
    create_dataset = TransformaParaWindowsDataset(sfreq)

    pipes = {}

    pipe_with_align = Pipeline([("Braindecode_dataset", create_dataset_with_align),
                                ("Net", clone(clf))])
    pipe = Pipeline([("Braindecode_dataset", create_dataset),
                     ("Net", clone(clf))])

    if args.ea == 'alignment':
        pipes["EEGNetv4_EA"] = pipe_with_align
    else:
        pipes["EEGNetv4_Without_EA"] = pipe

    # Define evaluation and train
    overwrite = False  # set to True if we want to overwrite cached results
    evaluation = EEGSharedEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        suffix=f"experiment_1_{args.dataset}",
        overwrite=overwrite,
        return_epochs=True,
        hdf5_path=run_dir,
        n_jobs=1,
        eval_config=eval_config,
        len_run=len_run,
        EA_in_eval=(args.ea == 'alignment'),
        online='on'
    )

    results = evaluation.process(pipes)
    print(results.head())

    # Save results
    results.to_csv(f"{run_dir}/{args.ea}_EEGNetShared_bn_{args.dataset}_{args.model}_online.csv")

    print("---------------------------------------")

    # return results


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)
