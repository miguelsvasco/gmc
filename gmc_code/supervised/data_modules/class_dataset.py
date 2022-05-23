import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from gmc_code.supervised.data_modules.extra.affect_dataset import AffectDataset


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
        # download
        if self.dataset == "mosei":
            train_data_file = os.path.join(self.data_dir, "mosei_train_a.dt")
            if not os.path.exists(train_data_file):
                raise RuntimeError('MOSEI Dataset not found.')

        elif self.dataset == "mosi":
            train_data_file = os.path.join(self.data_dir, "mosi_train_a.dt")
            if not os.path.exists(train_data_file):
                raise RuntimeError('MOSI Dataset not found.')
        else:
            raise ValueError(
                "[Classification Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")

    def setup(self, stage=None):

        if self.dataset == "mosei" or self.dataset == "mosi":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                alignment = 'a'

                # Train
                train_data_path = os.path.join(self.data_dir, self.dataset) + f'_train_{alignment}.dt'
                if not os.path.exists(train_data_path):
                    print(f"  - Creating new train data")
                    self.train_data = AffectDataset(self.data_dir, data=self.dataset, split_type='train')
                    torch.save(self.train_data, train_data_path)
                else:
                    print(f"  - Found cached train data")
                    self.train_data = torch.load(train_data_path)

                # Validation
                valid_data_path = os.path.join(self.data_dir, self.dataset) + f'_valid_{alignment}.dt'
                if not os.path.exists(valid_data_path):
                    print(f"  - Creating new valid data")
                    self.train_data = AffectDataset(self.data_dir, data=self.dataset, split_type='valid')
                    torch.save(self.train_data, valid_data_path)
                else:
                    print(f"  - Found cached valid data")
                    self.val_data = torch.load(valid_data_path)


            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:

                alignment = 'a'

                # Test
                test_data_path = os.path.join(self.data_dir, self.dataset) + f'_test_{alignment}.dt'
                if not os.path.exists(test_data_path):
                    print(f"  - Creating new test data")
                    self.test_data = AffectDataset(self.data_dir, data=self.dataset, split_type='test')
                    torch.save(self.test_data, test_data_path)
                else:
                    print(f"  - Found cached test data")
                    self.test_data = torch.load(test_data_path)


        else:
            raise ValueError(
                "[Classification Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")


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

        if self.dataset == "mosei":
            train_data_file = os.path.join(self.data_dir, "mosei_train_a.dt")
            if not os.path.exists(train_data_file):
                raise RuntimeError('MOSEI Dataset not found.')

        elif self.dataset == "mosi":
            train_data_file = os.path.join(self.data_dir, "mosi_train_a.dt")
            if not os.path.exists(train_data_file):
                raise RuntimeError('MOSI Dataset not found.')

        else:
            raise ValueError(
                "[Classification Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")


    def setup(self, stage=None):

        if self.dataset == "mosei" or self.dataset == "mosi":

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                alignment = 'a'

                # Train
                train_data_path = os.path.join(self.data_dir, self.dataset) + f'_train_{alignment}.dt'
                if not os.path.exists(train_data_path):
                    print(f"  - Creating new train data")
                    self.train_data = AffectDataset(self.data_dir, data=self.dataset, split_type='train')
                    torch.save(self.train_data, train_data_path)
                else:
                    print(f"  - Found cached train data")
                    self.train_data = torch.load(train_data_path)

                # Validation
                valid_data_path = os.path.join(self.data_dir, self.dataset) + f'_valid_{alignment}.dt'
                if not os.path.exists(valid_data_path):
                    print(f"  - Creating new valid data")
                    self.train_data = AffectDataset(self.data_dir, data=self.dataset, split_type='valid')
                    torch.save(self.train_data, valid_data_path)
                else:
                    print(f"  - Found cached valid data")
                    self.val_data = torch.load(valid_data_path)


            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:

                alignment = 'a'

                # Test
                test_data_path = os.path.join(self.data_dir, self.dataset) + f'_test_{alignment}.dt'
                if not os.path.exists(test_data_path):
                    print(f"  - Creating new test data")
                    self.test_data = AffectDataset(self.data_dir, data=self.dataset, split_type='test')
                    torch.save(self.test_data, test_data_path)
                else:
                    print(f"  - Found cached test data")
                    self.test_data = torch.load(test_data_path)
                
                self.set_dca_eval_sample_indices()
                self.partial_test_sampler = torch.utils.data.SubsetRandomSampler(
                    self.dca_partial_eval_indices
                )

        else:
            raise ValueError(
                "[Classification Dataset] Selected dataset: " + str(self.dataset) + " not implemented.")






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

