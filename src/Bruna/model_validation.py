from typing import Dict, List
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupKFold
from numpy import array
import copy
from braindecode.preprocessing import preprocess, Preprocessor
from alignment import euclidean_alignment


def increase_list(list_to_increase):
    lista_split = array(range(48))
    acc = []
    for item in list_to_increase:
        acc.append(lista_split + (48 * item))
    return array(acc).reshape(-1)


def split_train_test(
        windows_dataset, leave_one_subject_out=True
):
    """Convenient function for data division into training and
    testing. Performing one-subject-out and returning all the options as
    a list of indices.
    TODO: add option for K-fold or subject-wise K-fold.
    Returns indices of split data.
    Parameters
    ----------
    windows_dataset: WindowsDataset object from braindecode.
        This braindecode object need a description attribute with
        a column named subject.
    leave_one_subject_out: bool, default True
        If True, the split will be leave-one-subject-out.
    Returns
    -------
    split_ids: list of dict of indices for the train, test subsets.
        The size of the list is the number of folds.
        To access the train indices set, use split_idx[part]["train"],
        where part is the number of the fold.
    """
    ids_windows = windows_dataset.description.index.tolist()

    if leave_one_subject_out:
        print("Using K-Fold stratified in the indices/subject.")
        subj_ids = windows_dataset.description["subject"].tolist()

        leaver = LeaveOneGroupOut()
        splitter = leaver.split(ids_windows, groups=subj_ids)

    split_idx: List[Dict[str, list]] = [
        {"train": train_index, "test": test_index}
        for fold_id, (train_index, test_index) in
        enumerate(splitter)
    ]

    return split_idx


def split_train_val(DS_train, val_subj=None):
    """
    DS_train : BaseConcatDataset
    subj : int
    """
    DS = copy.deepcopy(DS_train)

    Aux_train_val = DS.description

    if val_subj is not None:
        Val_id = Aux_train_val[Aux_train_val['subject'] == val_subj]
        Aux_val_list = Val_id.index.tolist()
        Train_id = Aux_train_val[Aux_train_val['subject'] != val_subj]
        Aux_train_list = Train_id.index.tolist()

    else:
        Aux_train = Aux_train_val[Aux_train_val['session'] == 'session_T']
        Aux_train_list = Aux_train.index.tolist()
        Aux_val = Aux_train_val[Aux_train_val['session'] == 'session_E']
        Aux_val_list = Aux_val.index.tolist()

    Dic = {"train": Aux_train_list, "val": Aux_val_list}
    train_val = DS.split(Dic)
    Train = train_val['train']
    Val = train_val['val']
    return Train, Val


def split_dataset(w_dataset, subject_ids):
    """
    Split the data in subjects.
    Arguments:
    w_dataset : BaseConcatDataset
    subjects_id : list
    Return:
    Data_subj : BaseConcatDataset
    """
    list_ = []
    idx = subject_ids
    dataset_DS = copy.deepcopy(w_dataset)
    DF = dataset_DS.description

    for i in idx:
        Aux = DF[DF['subject'] == i]
        list_.append(Aux.index.tolist())

    Dic = {f'{idx[i]}': list_[i] for i in range(len(idx))}
    Data_subj = dataset_DS.split(Dic)

    return Data_subj


def EA_dataset(Data_subj, subject_ids):
    """
    Align each subject in your own space
    Arguments:
    Data_subj : BaseConcatDataset
    subject_ids : list
    session : boolean
    Return:
    Data_subj_EA : BaseConcatDataset
    """

    idx = subject_ids
    Data_subj_EA = copy.deepcopy(Data_subj)

    for i in idx:
        Data_subj_EA[f'{i}'] = preprocess(Data_subj[f'{i}'],
                                          [Preprocessor(euclidean_alignment, apply_on_array=True)])
    return Data_subj_EA


def split_run(Data_subj, run):
    """
    Split each subject in runs
    Arguments:
    Data_subj : BaseConcatDataset
      Data from ONE subject
    test_subj : int
    subject_ids : list
    run : list of int
    Return:
    Data_subj_run : BaseConcatDataset
      Dataset from one subject splited in runs
    """
    dataset_DS = copy.deepcopy(Data_subj)
    DF = dataset_DS.description

    run_subj = []

    # Take each run and add to list
    for r in run:
        DS_run = DF[DF['run'] == f'run_{r}']
        run_subj.append(DS_run.index.tolist())

    # Transform into Dic and split BaseConcatDataset
    Dic = {f'run_{run[i]}': run_subj[i] for i in range(len(run))}
    Data_subj_run = dataset_DS.split(Dic)

    return Data_subj_run


def split_size(Data_subj_run, run):
    """
    Split a specific subject in different sizes
    Arguments:
      Data_subj_run : BaseConcatDataset
        Data from ONE subject already splitted in runs
      run : list of int
    Return:
      Data_runs : list
        list of BaseConcatDatasets with different number of elements
    """
    # In this list, we'll save BaseConcatDatasets
    Data_runs = []

    # Create different tests data with different sizes each
    for i in range(len(run)):
        Aux = []
        for j in range(i + 1):
            Aux.append(Data_subj_run[f'run_{j}'])

        DS = BaseConcatDataset(Aux)
        Data_runs.append(DS)

    Data_runs = array(Data_runs, dtype=object)
    return Data_runs