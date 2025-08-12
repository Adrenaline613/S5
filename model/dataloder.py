import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import KFold
from glob import glob
import os

from model.augmentation import *


def split_dataset(
        data_dir: str,
        split_ratio,
        test: bool = False,
        mode: str = 'fixed',
        sub_dir: list = None,
        shuffle: bool = False) -> list:

    """
    Split the dataset into training, validation and testing set

    Args:
        data_dir (str): the path of the dataset

        split_ratio (float):
            the ratio of training, validating, and testing set, for example, [0.6, 0.2, 0.2] or [4, 1, 1], when mode is
            'sub_dir', split_ratio is not used, default None

        test (bool):
            whether to split the testing set, if True, split the dataset into training, validating, and testing set,
            else split the dataset into training and validating set, default False

        mode (str): the mode of splitting the dataset.
            'fixed':
                use the fixed ratio to split the dataset into training and testing set (only one fold),
                the ratio is defined by split_ratio
            'k_fold':
                use k-fold cross validation to split the dataset into training and testing set,
                the number of folds is defined by split_ratio
            'sub_dir':
                Read data from the training and validation subfolders to construct the training and validation sets.
                At this point, data_dir should be the upper level folder of the subfolders, for example:
                data_dir = 'C:/dataset/DREAMS/DatabaseSpindles' sub_dir = ['train', 'val', 'test']

        sub_dir (list): the list of subfolders name, default None, only used when mode is 'sub_dir'

        shuffle (bool): whether to shuffle the dataset, default False, only used when mode is 'fixed' or 'k_fold'

    Returns:
        list: the path of training and testing set
    """

    assert mode in ['fixed', 'k_fold', 'sub_dir'], 'mode should be fixed, k_fold or sub_dir'
    if test:
        if mode == 'sub_dir':
            assert len(sub_dir) == 3, 'sub_dir should contain 3 elements'
        else:
            assert len(split_ratio) == 3 and all(
                [i > 0 for i in split_ratio]), 'split_ratio should contain 3 elements and all greater than 0'
    else:
        if mode == 'sub_dir':
            assert len(sub_dir) == 2, 'sub_dir should contain 2 elements'
        else:
            assert len(split_ratio) == 2 and all(
                [i > 0 for i in split_ratio]), 'split_ratio should contain 2 elements and all greater than 0'

    k_fold_path = []

    if mode == 'sub_dir':
        all_data_dir = []
        for subdir in sub_dir:
            all_data_dir.append(np.array(glob(os.path.join(data_dir, subdir, '*.npz'))))
        k_fold_path.append(all_data_dir)

    else:
        split_ratio = np.array(split_ratio)
        split_ratio = split_ratio / split_ratio.sum()

        all_data_dir = np.array(glob(os.path.join(data_dir, '*.npz')))

        if shuffle:
            np.random.shuffle(all_data_dir)
        else:
            np.sort(all_data_dir)

        if mode == 'fixed':
            train_index = round(len(all_data_dir) * split_ratio[0])
            train_path = all_data_dir[:train_index]

            if test:
                val_index = train_index + round(len(all_data_dir) * split_ratio[1])
                val_path = all_data_dir[train_index:val_index]
                test_path = all_data_dir[val_index:]
                k_fold_path.append([train_path, val_path, test_path])

            else:
                val_path = all_data_dir[train_index:]
                k_fold_path.append([train_path, val_path])

        else:
            num_fold = int(split_ratio.sum() / split_ratio[-1])
            if test:
                val_num = int(len(all_data_dir) * split_ratio[1])
                kf = KFold(n_splits=num_fold, shuffle=False)
                for train_index, test_index in kf.split(all_data_dir):
                    np.random.shuffle(train_index)
                    val_path = all_data_dir[train_index[:val_num]]
                    train_path = all_data_dir[train_index[val_num:]]
                    test_path = all_data_dir[test_index]
                    k_fold_path.append([train_path, val_path, test_path])
            else:
                kf = KFold(n_splits=num_fold, shuffle=False)
                for train_index, val_index in kf.split(all_data_dir):
                    train_path = all_data_dir[train_index]
                    val_path = all_data_dir[val_index]
                    k_fold_path.append([train_path, val_path])

    return k_fold_path


class Load_Dataset_sp(Dataset):
    """
    A custom Dataset class to load data from multiple numpy files.

    Args:
        np_dataset (list): A list of paths to numpy files containing the dataset.

    """

    def __init__(self, np_dataset):
        super(Load_Dataset_sp, self).__init__()

        for i in range(len(np_dataset)):
            x = np.load(np_dataset[i])["segments"]
            y = np.load(np_dataset[i])["labels"]
            print(f"\033[31m\rLoading {i+1}/{len(np_dataset)}. {np_dataset[i]}: data{x.shape}, label{y.shape}\033[0m",
                  end='', flush=True)
            if not hasattr(self, 'x_data'):
                self.x_data = x
                self.y_data = y
            else:
                self.x_data = np.concatenate((self.x_data, x), axis=0)
                self.y_data = np.concatenate((self.y_data, y), axis=0)

        # Set the length of the dataset
        self.len = self.x_data.shape[0]

        # Convert numpy arrays to torch tensors
        self.x_data = torch.from_numpy(self.x_data.astype(np.float32))
        self.y_data = torch.from_numpy(self.y_data.astype(np.int8)).long()

        # Adjust the shape of x_data if necessary
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(1, 0, 2)
        else:
            self.x_data = self.x_data.unsqueeze(1)

        print(f"\033[32m\rData loaded, Subject num: {len(np_dataset)}, Epoch num: {self.len}\033[0m")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Load_Dataset_cl(Dataset):
    """
    A custom Dataset class to load contrastive learning dataset from multiple numpy files.

    Args:
        np_dataset (list): A list of paths to numpy files containing the dataset.

    """

    def __init__(self, np_dataset):
        super(Load_Dataset_cl, self).__init__()

        for i in range(len(np_dataset)):

            x = np.load(np_dataset[i])["segments"]

            print(f"\033[31m\rLoading {i+1}/{len(np_dataset)}. {np_dataset[i]}: {x.shape}\033[0m", end='', flush=True)

            if not hasattr(self, 'x_train'):
                self.x_train = x
            else:
                self.x_train = np.concatenate((self.x_train, x), axis=0)

        self.x_train = np.expand_dims(self.x_train.astype(np.float32), axis=1)

        self.len = self.x_train.shape[0]

        # applying the transform
        self.transform = Compose(
            transforms=[
                SpindleBandScale(),
                BandStop(),
                AdditiveNoise(),
                AmplitudeScale(),
                TimeShift(),
                ZeroMask()
            ]
        )
        self.two_transform = TwoTransform(self.transform)

        print(f"\033[32m\rData loaded, Subject num: {len(np_dataset)}, Epoch num: {self.len}\033[0m")

    def __getitem__(self, index):
        input_a, input_b = self.two_transform(self.x_train[index])
        input_a = torch.from_numpy(input_a).float()
        input_b = torch.from_numpy(input_b).float()
        inputs = [input_a, input_b]
        return inputs

    def __len__(self):
        return self.len


def make_dataset(train_files, val_files, test_files=None, training_mode='pretrain'):
    """
    Create datasets for training, validation, and optionally testing.

    Args:
        train_files (list): List of paths to training data files.
        val_files (list): List of paths to validation data files.
        test_files (list, optional): List of paths to testing data files. Default is None.
        training_mode (str): The mode of training, if 'pretrain', use Load_Dataset_cl, else use Load_Dataset_sp.

    Returns:
        tuple: A tuple containing the training and validation datasets, and optionally the testing dataset.

    """

    load_from_numpy = Load_Dataset_cl if training_mode == 'pretrain' else Load_Dataset_sp

    train_dataset = load_from_numpy(train_files)
    val_dataset = load_from_numpy(val_files)
    if test_files is None:
        return train_dataset, val_dataset
    test_dataset = load_from_numpy(test_files)

    return train_dataset, val_dataset, test_dataset


def make_dataloader(train_dataset, val_dataset, batch_size=128, test_dataset=None):
    """
    Create dataloaders for training, validation, and optionally testing.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        batch_size (int): The number of samples per batch.
        test_dataset (Dataset, optional): The testing dataset. Default is None.

    Returns:
        tuple: A tuple containing the training and validation dataloaders, and optionally the testing dataloader.
    """
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    if test_dataset is None:
        return train_loader, val_loader

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_loader, val_loader, test_loader
