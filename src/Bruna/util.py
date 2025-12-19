"""Util functions."""

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import random
import warnings
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import skorch
import torch

from braindecode.datasets import BaseConcatDataset
import copy
from braindecode.preprocessing import (preprocess, Preprocessor)
from alignment import euclidean_alignment


_seed = None
_flag_deterministic = torch.backends.cudnn.deterministic
_flag_cudnn_benchmark = torch.backends.cudnn.benchmark
MAX_SEED = (
        np.iinfo(np.uint32).max + 1
)  # 2**32, the actual seed should be in [0, MAX_SEED - 1] for uint32


def set_determinism(
        seed: Optional[int] = np.iinfo(np.uint32).max,
        use_deterministic_algorithms: Optional[bool] = None,
) -> None:
    """Set random seed for modules to enable or disable deterministic training.

    Args:
        seed: the random seed to use, default is np.iinfo(np.int32).max.
            It is recommended to set a large seed, i.e. a number that
            has a good balance
            of 0 and 1 bits. Avoid having many 0 bits in the seed.
            if set to None, will disable deterministic training.
        use_deterministic_algorithms: Set whether PyTorch operations must
        use "deterministic" algorithms.
        additional_settings: additional settings that need to set random seed.
    """
    if seed is None:
        # cast to 32 bit seed for CUDA
        seed_ = torch.default_generator.seed() % (np.iinfo(np.int32).max + 1)
        torch.manual_seed(seed_)
    else:
        seed = int(seed) % MAX_SEED
        torch.manual_seed(seed)

    global _seed
    _seed = seed
    random.seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.flags_frozen():
        warnings.warn(
            "PyTorch global flag support of backends is disabled, "
            + "enable it to set global `cudnn` flags."
        )
        torch.backends.__allow_nonbracketed_mutation_flag = True

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    else:  # restore the original flags
        torch.backends.cudnn.deterministic = _flag_deterministic
        torch.backends.cudnn.benchmark = _flag_cudnn_benchmark
    if use_deterministic_algorithms is not None:
        if hasattr(
                torch, "use_deterministic_algorithms"
        ):  # `use_deterministic_algorithms` is new in torch 1.8.0
            torch.use_deterministic_algorithms(use_deterministic_algorithms)
        elif hasattr(
                torch, "set_deterministic"
        ):  # `set_deterministic` is new in torch 1.7.0
            torch.set_deterministic(use_deterministic_algorithms)
        else:
            warnings.warn(
                "use_deterministic_algorithms=True, but "
                + "PyTorch version is too old to set the mode."
            )


def parse_args():
    """
    Parse function.
    Mixing args + config file.
    Returns
    -------

    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default="config/config.yaml",
        help="Path to config file with all the training parameters needed",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="select the dataset to be used on the analyse.",
        default="BNCI2014001",
        choices=["BNCI2014001", "PhysionetMI", "Cho2017", "Shin2017A", "Schirrmeister2017"],
    )

    parser.add_argument(
        "--num_exp",
        type=str,
        help="select which experiment we gonna run.",
        default="exp_1",
        choices=["exp_1", "exp_2", "exp_3", "exp_4", "exp_5", "exp_hybrid", "exp_shared"],
    )

    parser.add_argument(
        "--ea",
        type=str,
        help="select alignment",
        default="alignment",
        choices=["alignment", "no-alignment"],
    )

    parser.add_argument(
        "--freeze",
        type=str,
        help="select whether to freeze shared layers in evaluation.",
        default="no-freeze",
        choices=["freeze", "no-freeze"],
    )
        
    parser.add_argument(
        "--session",
        type=str,
        help="select session",
        default="both",
        choices=["both", "session_T", "session_E"],
    )

    parser.add_argument(
        "--online",
        type=int,
        help="select online or offline",
        default=0,
        choices=[0, 1],
    )

    parser.add_argument(
        "--eval_config_file",
        type=str,
        default="config/eval_config.yaml",
        help="Path to config file with all the evaluation parameters needed (only for hybrid domain adaptation)",
        choices=["config/eval_config.yaml", "config/eval_config_shared.yaml"],
    )

    parser.add_argument(
        "--model",
        type=str,
        help="select model",
        default="EEGNet",
        choices=["EEGNet", "DeepNet", "ShallowNet", "DeepNetShared", "ShallowNetShared", "EEGNetShared", "EEGNetNormTest"],
    )

    parser.add_argument(
        "--remove_bn",
        type=str,
        default='False',
        help="To remove batchnormalization layers",
        choices=['True', 'False', 'One-bn', 'LEA', 'One-LEA'],
    )

    parser.add_argument(
        "--criterion_type",
        type=str,
        help="Select a loss function",
        default="AlignmentLoss",
        choices=["AlignmentLoss", "NLLLoss"],
    )

    parser.add_argument(
        "--mode",
        type=str,
        help="select mode of the evaluation",
        default="Fit",
        choices=["Fit", "Inference"],
    )

        
    args = parser.parse_args()
    return args


def set_run_dir(config, args):
    """
    Set the run directory.
    Parameters
    ----------
    config: dict
        Dictionary with all the parameters needed for the training.
    args: args from argparse
        Arguments from argparse.

    Returns
    -------
    run_dir: Path
    Path to the run directory.
    experiment_name: str
    Name of the experiment.

    """
    output_dir = Path(config.train.run_report + "/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    experiment_name = (
            config.train.experiment_name
            + "-"
            + str(args.dataset)
            + "-"
            + str(args.ea)
            + '-'
            + str(args.freeze)
            + '-'
            + str(args.num_exp)
            + '-'
            + str(args.model)
            + '-'
            + str(args.session)

    )

    run_dir = output_dir / (experiment_name)
    print(f"The run_dir is {run_dir}")
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        pass
    else:
        print("First time running, creating folder")
        run_dir.mkdir(exist_ok=True, parents=True)

    # Create savedir
    print(
        f"The experiment name in Mlflow will be:{experiment_name}", end="\n",
    )
    return run_dir, experiment_name


def starting_mlflow(config, args, baseline=False, model_name="", task=""):
    """
    Util function to starting the active_run with mlflow.
    Parameters
    ----------
    config
    args
    baseline

    Returns
    -------

    """
    if baseline:
        experiment_name = (
                config.train.experiment_name + "-" + model_name + "-" + task
        )
    else:
        experiment_name = (
                config.train.experiment_name
                + "-"
                + str(args.dataset)
                + "-"
                + str(args.ea)
        )

    mlflow.set_experiment(experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    active_run = mlflow.start_run(
        run_name=config.train.mlflow_dir,
        experiment_id=experiment.experiment_id,
    )

    return active_run


class TensorBoardCallback(skorch.callbacks.TensorBoard):
    """Tensorboard as skorch callback.

    Link: https://github.com/NINFA-UFES/ESPset/blob/
    94660d6c19d9006ea7a166309eb9089609d4e0c6/networks.py
    """

    def on_train_end(self, net, X=None, y=None, **kwargs):
        neural_net_params = {
            k: v
            for k, v in net.get_params().items()
            if type(v) in (int, float, str, bool)
        }
        neural_net_params["NeuralNet class"] = net.__class__.__name__
        if "valid_loss" in net.history[-1]:
            loss_name = "valid_loss"
        else:
            loss_name = "train_loss"
        best_loss = min(net.history[:, loss_name])
        self.writer.add_hparams(
            neural_net_params, {"hparam/best_loss": best_loss}
        )


def log_mlflow(active_run, model, run_report, config, args, split_ids):
    """Log model and performance on Mlflow system."""

    with active_run:
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        for key, value in vars(config).items():
            mlflow.log_param(key, value)

        for key, value in split_ids.items():
            mlflow.log_param(key, value)

        try:

            for key, value in vars(split_ids).items():
                mlflow.log_param(key, value)

            mlflow.log_artifacts(str(run_report), artifact_path="events")
            raw_model = model.module if hasattr(model, "module") else model
            mlflow.pytorch.log_model(raw_model, "final_model")
        except Exception as ex:
            print(f"Error {ex}, ignore if train option.")

            print("log the model fail in option 1, works in option 2.")


