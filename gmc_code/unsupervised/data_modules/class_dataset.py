import os
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from gmc_code.unsupervised.data_modules.extra.mhd_dataset import MHDDataset


class ClassificationDataModule(LightningDataModule):
    def __init__(self, dataset, data_dir, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_config = data_config

        # Data-specific variables - fill with setup function;
        self.transform = None
        self.train_data, self.val_data, self.test_data = None, None, None

    def prepare_data(self):

        if self.dataset == "mhd":
            train_data_file = os.path.join(self.data_dir, "mhd_train.pt")
            test_data_file = os.path.join(self.data_dir, "mhd_test.pt")

            if not os.path.exists(train_data_file) or not os.path.exists(test_data_file):
                raise RuntimeError('MHD Dataset not found. Please generate dataset and place it in the data folder.')
        else:
            raise ValueError(
                "[Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")

    def setup(self, stage=None):

        # Setup Dataset:
        if self.dataset == "mhd":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                mhd_full = MHDDataset(self.data_dir, train=True)
                self.train_data, self.val_data = random_split(mhd_full,
                                                              [int(0.9*len(mhd_full)),
                                                               len(mhd_full) - int(0.9*len(mhd_full))])
                self.dims = tuple(self.train_data[0][0].shape)

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_data = MHDDataset(self.data_dir, train=False)
                self.dims = tuple(self.test_data[0][0].shape)

        else:
            raise ValueError(
                "[Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")


    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
        )


class DCADataModule(LightningDataModule):
    def __init__(self, dataset, data_dir, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = dataset
        self.data_dir = data_dir
        self.data_config = data_config

        # Data-specific variables - fill with setup function;
        self.transform = None
        self.train_data, self.val_data, self.test_data = None, None, None
        self.test_sampler = None
        self.dca_partial_eval_indices = None

    def set_dca_eval_sample_indices(self):
        if self.dca_partial_eval_indices is None:
            self.dca_partial_eval_indices = np.random.choice(
                list(range(len(self.test_data))),
                self.data_config["n_dca_samples"],
                replace=False,
            )

    def prepare_data(self):
        # download
        if self.dataset == "mhd":
            train_data_file = os.path.join(self.data_dir, "mhd_train.pt")
            test_data_file = os.path.join(self.data_dir, "mhd_test.pt")

            if not os.path.exists(train_data_file) or not os.path.exists(test_data_file):
                raise RuntimeError('MHD Dataset not found. Please generate dataset and place it in the data folder.')

        else:
            raise ValueError(
                "[DCA Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")


    def setup(self, stage=None):

        # Setup Dataset:
        if self.dataset == "mhd":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                mhd_full = MHDDataset(self.data_dir, train=True)
                self.train_data, self.val_data = random_split(mhd_full,
                                                              [int(0.9*len(mhd_full)),
                                                               len(mhd_full) - int(0.9*len(mhd_full))])
                self.dims = tuple(self.train_data[0][0].shape)

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.test_data = MHDDataset(self.data_dir, train=False)
                self.dims = tuple(self.test_data[0][0].shape)

                self.set_dca_eval_sample_indices()
                self.partial_test_sampler = torch.utils.data.SubsetRandomSampler(
                    self.dca_partial_eval_indices
                )

        else:
            raise ValueError(
                "[DCA Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")



    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            sampler=self.partial_test_sampler,
            drop_last=False,
        )

