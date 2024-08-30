import numpy as np
from pathlib import Path
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch_geometric.loader import DataLoader as GeomDataLoader
from torch.utils.data import random_split, Subset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .build import *
from utils.misc import worker_init_fn


class PartNetDataModule(LightningDataModule):
    """
    Data module for loading PartNet dataset using Pytorch Lightning.

    :param args: Arguments received from command line.
    :param config: Configuration object with dataset parameters. 
    """

    def __init__(self, args, cfg):
        super().__init__()
        self.args = args
        self.cfg = cfg

    def setup(self, stage: str = 'fit') -> None:
        """
        Setup the PartNet dataset for training. This method is called to initialize the datasets.

        :param stage: Stage for which the datasets are being prepared; ('fit' for training/validation, 'test' for testing).
        """
        # assign train/val/test dataset for use in dataloaders
        if stage == 'fit':
            partnet = build_dataset_from_cfg(self.cfg.dataset.train, self.cfg.dataset.train.others)
            # split into train and val
            train_size = int(0.8 * partnet.__len__())
            val_size = partnet.__len__() - train_size
            self.partnet_train, self.partnet_val = random_split(partnet, [train_size, val_size])
        elif stage == 'test':
            raise NotImplementedError('TEST dataset not implemented yet')
        elif stage == 'predict':
            # NOTE Check is we want to do this, a.k.a. take the whole dataset. 
            self.partnet_predict = build_dataset_from_cfg(self.cfg.dataset.train, self.cfg.dataset.train.others)
        else:
            raise NotImplementedError(f'Available stages are: [fit, test, predict], you passed {stage}')

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        :return: DataLoader for the training dataset.
        """

        return DataLoader(
            dataset=self.partnet_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=64,  # there are 64 cpu's on the remote machine
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        :return: DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.partnet_val,
            batch_size=self.cfg.batch_size,
            num_workers=64,  # there are 64 cpu's on the remote machine
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.partnet_predict,
            batch_size=self.cfg.batch_size,
            num_workers=64,  # there are 64 cpu's on the remote machine
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )


class CeasarDataModule(LightningDataModule):
    """
    Data module for loading Ceasar dataset using Pytorch Lightning.

    :param args: Arguments received from command line.
    :param config: Configuration object with dataset parameters. 
    """

    def __init__(self, args, cfg):
        super().__init__()
        self.args = args
        self.cfg = cfg

    def setup(self, stage: str = 'fit') -> None:
        """
        Setup the PartNet dataset for training. This method is called to initialize the datasets.

        :param stage: Stage for which the datasets are being prepared; ('fit' for training/validation, 'test' for testing).
        """
        # assign train/val/test dataset for use in dataloaders
        if stage == 'fit':
            ceasar = build_dataset_from_cfg(self.cfg.dataset.train, self.cfg.dataset.train.others)
            # split into train and val
            train_size = int(0.8 * ceasar.__len__())
            val_size = ceasar.__len__() - train_size
            self.ceasar_train, self.ceasar_val = random_split(ceasar, [train_size, val_size])
        elif stage == 'test':
            raise NotImplementedError('TEST dataset not implemented yet')
        elif stage == 'predict':
            # NOTE Check is we want to do this, a.k.a. take the whole dataset. 
            self.ceasar_predict = build_dataset_from_cfg(self.cfg.dataset.train, self.cfg.dataset.train.others)
        else:
            raise NotImplementedError(f'Available stages are: [fit, test, predict], you passed {stage}')

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        :return: DataLoader for the training dataset.
        """

        return DataLoader(
            dataset=self.ceasar_train,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=64,  # there are 64 cpu's on the remote machine
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        :return: DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.ceasar_val,
            batch_size=self.cfg.batch_size,
            num_workers=64,  # there are 64 cpu's on the remote machine
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.ceasar_predict,
            batch_size=self.cfg.batch_size,
            num_workers=64,  # there are 64 cpu's on the remote machine
            worker_init_fn=worker_init_fn,
            pin_memory=True
        )


class GNNDataModule(LightningDataModule):
    def __init__(self, args, cfg, k_fold: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.k_fold = k_fold
        self.dataset_percentage = getattr(cfg, 'dataset_percentage', 100) / 100.0  # Default to 100% if not specified
        self.full_dataset = build_dataset_from_cfg(
            self.cfg.dataset.train) if not self.k_fold else []  # in case of k_fold we dont need the full dataset

        if not self.k_fold:
            train_size = int(0.70 * self.full_dataset.__len__())
            val_size = int(0.20 * self.full_dataset.__len__())
            test_size = self.full_dataset.__len__() - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.full_dataset,
                                                                                   [train_size, val_size, test_size])

            print(f"DATA SPLIT: {train_size} train, {val_size} val, {test_size} test samples")

    def setup_kfold(self, fold=None) -> None:
        """
        Load the premade fold from the disk based on the "fold" parameter.
        """

        available_folds = len(
            [f for f in Path(self.cfg.dataset.train.DATA_PATH).iterdir() if f.is_dir() and f.name.startswith('fold_')])
        if fold > available_folds:
            raise ValueError(f"There are {available_folds} fold available. You tried to setup fold {fold + 1}")

        # specify the paths where the folds are stored
        train_data_path = Path(self.cfg.dataset.train.DATA_PATH) / f'fold_{fold}/training/'
        val_data_path = Path(self.cfg.dataset.train.DATA_PATH) / f'fold_{fold}/validation/'
        test_data_path = Path(self.cfg.dataset.test.DATA_PATH)

        # load train, val test
        train_dataset = build_dataset_from_path(self.cfg.dataset.train, train_data_path)
        item = next(iter(train_dataset))

        self.val_dataset = build_dataset_from_path(self.cfg.dataset.train, val_data_path)
        self.test_dataset = build_dataset_from_path(self.cfg.dataset.train, test_data_path)

        # shrink train data based on "dataset_percentage"
        total_size = int(self.dataset_percentage * train_dataset.__len__())
        self.train_dataset = Subset(train_dataset, range(total_size))

        print("TRAIN: ", len(self.train_dataset))
        print("VAL: ", len(self.val_dataset))
        print("TEST: ", len(self.test_dataset))

    def get_leaf_labels(self):
        # Collect all labels, then filter
        all_labels = []
        for i in range(len(self.full_dataset)):
            graph_data = self.full_dataset[i]
            all_labels.extend(graph_data.y.tolist())

        # Convert list to numpy array and filter for leaf nodes
        all_labels = np.array(all_labels)
        leaf_labels = all_labels[all_labels < 22]
        return leaf_labels

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        :return: DataLoader for the training dataset.
        """
        return GeomDataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=64,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        :return: DataLoader for the validation dataset.
        """
        return GeomDataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=64,
            pin_memory=True
        )

    def test_dataloader(self):
        """
        Create a DataLoader for the test dataset.

        :return: DataLoader for the test dataset.
        """
        return GeomDataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=64,
            pin_memory=True
        )


class PartNetEmbeddingsDataModule(LightningDataModule):
    def __init__(self, args, cfg, k_fold: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.k_fold = k_fold
        self.dataset_percentage = getattr(cfg, 'dataset_percentage', 100) / 100.0  # Default to 100% if not specified
        self.full_dataset = build_dataset_from_cfg(self.cfg.dataset.train) if not self.k_fold else []

        if not self.k_fold:
            train_size = int(0.70 * len(self.full_dataset))
            val_size = int(0.20 * len(self.full_dataset))
            test_size = len(self.full_dataset) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.full_dataset,
                                                                                   [train_size, val_size, test_size])

            print(f"DATA SPLIT: {train_size} train, {val_size} val, {test_size} test samples")

    def setup_kfold(self, fold=None) -> None:
        """
        Load the premade fold from the disk based on the "fold" parameter.
        """
        available_folds = len(
            [f for f in Path(self.cfg.dataset.train.DATA_PATH).iterdir() if f.is_dir() and f.name.startswith('fold_')])
        if fold > available_folds:
            raise ValueError(f"There are {available_folds} fold available. You tried to setup fold {fold + 1}")

        # specify the paths where the folds are stored
        train_data_path = Path(self.cfg.dataset.train.DATA_PATH) / f'fold_{fold}/training/'
        val_data_path = Path(self.cfg.dataset.train.DATA_PATH) / f'fold_{fold}/validation/'
        test_data_path = Path(self.cfg.dataset.test.DATA_PATH)

        # load train, val test
        train_dataset = build_dataset_from_path(self.cfg.dataset.train, train_data_path)
        self.val_dataset = build_dataset_from_path(self.cfg.dataset.train, val_data_path)
        self.test_dataset = build_dataset_from_path(self.cfg.dataset.train, test_data_path)

        # shrink train data based on "dataset_percentage"
        total_size = int(self.dataset_percentage * len(train_dataset))
        self.train_dataset = Subset(train_dataset, range(total_size))

        print("TRAIN: ", len(self.train_dataset))
        print("VAL: ", len(self.val_dataset))
        print("TEST: ", len(self.test_dataset))

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        :return: DataLoader for the training dataset.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=64,
            pin_memory=True
        )

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        :return: DataLoader for the validation dataset.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=64,
            pin_memory=True
        )

    def test_dataloader(self):
        """
        Create a DataLoader for the test dataset.

        :return: DataLoader for the test dataset.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=64,
            pin_memory=True
        )
