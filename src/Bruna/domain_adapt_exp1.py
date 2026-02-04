import warnings

import braindecode
import torch
import moabb
from joblib import parallel_backend
from moabb.datasets import BNCI2014_001, Weibo2014, Shin2017A, Schirrmeister2017, PhysionetMI
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

"""
For the hybrid/combined model
"""

moabb.set_log_level("info")
warnings.filterwarnings("ignore")

def main(args):
    """
    Parameters
    ----------
    args : object
    """
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

    if args.dataset == 'BNCI2014001':
        dataset = BNCI2014_001()
        ch=None
        subjects = dataset.subject_list
    elif args.dataset == 'Weibo2014':
        dataset = Weibo2014()
        ch = ["FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3",
              "CP1", "CPz", "CP6", "CP4", "CP2"]
    elif args.dataset == 'Shin2017A':
        dataset = Shin2017A(accept=True)
        ch = None

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

    events = ["right_hand", "left_hand"]

    paradigm = MotorImagery_(events=events, n_classes=len(events), channels=ch)

    datasets = [dataset]
    events = ["left_hand", "right_hand"]
    n_classes = len(events)

    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[2])
    n_chans = X.shape[1]
    input_window_samples = X.shape[2]
    rpc = len(meta['session'].unique()) * len(meta['run'].unique())

    print(f"(2) Get Data Done {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    num_subjects = len(dataset.subject_list)

    model = HybridModel(num_subjects - 1, args.model, n_chans, n_classes, input_window_samples, config=config,
                        freeze=args.freeze, args=args)
    # Send model to GPU

    if cuda:
        model.cuda()
    torchinfo.summary(model, input_size=(config.train.batch_size, X[0].shape[0] * (num_subjects - 1), X[0].shape[1]))

    # Create Classifier
    print(args)
    criterion_type = args.criterion_type  # Define the loss function to use
    clf = define_hybrid_clf(model, config, experiment_name, criterion_type)

    print(f"(3) Created clf {(time() - init_time) * 1000}ms | {(time() - init_time)}s")
    print(f"Type of loss function: {criterion_type}")

    runs = meta.run.values
    sessions = meta.session.values
    one_session = sessions == np.unique(sessions)[0]
    one_run = runs == np.unique(runs)[0]
    run_session = np.logical_and(one_session, one_run)
    len_run = sum(run_session * 1) if dataset.code == '001-2014' else config.train.len_run

    hybrid_adapter = HybridAggregateTransform(data_code=dataset.code)
    hybrid_adapter_EA = HybridAggregateTransform(EA_len_run=len_run, data_code=dataset.code)

    pipes = {}

    pipe_with_align = Pipeline([("Hybrid_adapter", hybrid_adapter_EA),
                                ("Net", clone(clf))])
    pipe = Pipeline([("Hybrid_adapter", hybrid_adapter),
                     ("Net", clone(clf))])

    freeze_tag = ["-Frozen", "-Not_Frozen"][args.freeze == "no-freeze"]

    if args.ea == 'alignment':
        pipes[args.model + "|_EA" + freeze_tag] = pipe_with_align
    else:
        pipes[args.model + "|_Without_EA" + freeze_tag] = pipe

    print(f"(4) Before eval setup {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    eval_config.train.experiment_name = config.train.experiment_name

    # Define evaluation and train
    overwrite = True  # set to True if we want to overwrite cached results
    evaluation = HybridEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        suffix=f"experiment_1_{args.dataset}",
        overwrite=overwrite,
        return_epochs=True,
        hdf5_path=run_dir,
        n_jobs=1,
        eval_config=eval_config,
        EA_in_eval=(args.ea == 'alignment'),
        len_run=len_run,
        wandb_params=(args, config),
        run_dir=run_dir,
        mode=args.mode,
        remove_bn=args.remove_bn,
        criterion_type = args.criterion_type,
    )

    print(f"(5) Before eval {(time() - init_time) * 1000}ms | {(time() - init_time)}s")

    if args.remove_bn=='True':
        bn = 'nobn'
    else:
        bn = 'bn'

    print(f"(6) Remove batch normalization? {args.remove_bn} - {bn}")

    #with parallel_backend("sequential"):
    #    results = evaluation.process(pipes)
    results = evaluation.process(pipes)
    print(results.head())

    # Save results
    print(run_dir)
    print(experiment_name)
    #print(f"{run_dir}/Heads-shared_{experiment_name}_{args.remove_bn}_{criterion_type}_{args.mode}_results.csv")
    results.to_csv(f"{run_dir}/Heads-shared_{experiment_name}_{args.remove_bn}_{criterion_type}_{args.mode}_results.csv")
    #results.to_csv(f"{run_dir}/Test-refactoring_{experiment_name}_{args.remove_bn}_{criterion_type}_{args.mode}_results.csv")

    print("---------------------------------------")

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    args = parse_args()
    main(args)