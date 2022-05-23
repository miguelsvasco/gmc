import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


# 'state', 'action', 'next_state', 'reward', 'terminal'
def generate_dataset_filename(scenario, scenario_cfg):
    if scenario == 'pendulum':

        snd_f = scenario_cfg['sound_frequency']
        snd_vel = scenario_cfg['sound_velocity']
        snd_rcv = scenario_cfg['sound_receivers']

        train_filename = '_'.join([
            f'train_pendulum_dataset_samples{scenario_cfg["train_samples"]}', f'stack{scenario_cfg["n_stack"]}',
            f'freq{snd_f}', f'vel{snd_vel}',
            f'rec{str(snd_rcv)}.pt'])

        test_filename = '_'.join([
            f'test_pendulum_dataset_samples{scenario_cfg["test_samples"]}', f'stack{scenario_cfg["n_stack"]}',
            f'freq{snd_f}', f'vel{snd_vel}',
            f'rec{str(snd_rcv)}.pt'])

    else:
        raise ValueError('Incorrect initialization of Atari Game Scenario.')

    return train_filename, test_filename

class AtariDataset(Dataset):
    def __init__(self,
                 scenario,
                 data_dir,
                 scenario_cfg,
                 train=True):

        # Get variables
        self.scenario = scenario
        self.data_dir = data_dir
        self.scenario_cfg = scenario_cfg
        train_dataset_filename, test_dataset_filename = generate_dataset_filename(scenario=self.scenario, scenario_cfg=self.scenario_cfg)

        # Loading data from dataset
        if train:
            self._image_states, self._sound_states, self._actions, self._rewards,\
            self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, train_dataset_filename))
        else:
            self._image_states, self._sound_states, self._actions, self._rewards, \
            self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, test_dataset_filename))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, sound)
        """
        img, sound, action, reward, next_img, next_snd, done = self._image_states[index], self._sound_states[index],\
                                                               self._actions[index], self._rewards[index], \
                                                               self._next_image_states[index], self._next_sound_states[index], self._done[index]

        return [img.squeeze(), sound.squeeze()], [next_img.squeeze(), next_snd.squeeze()], action.squeeze(), reward.squeeze(), done.squeeze()

    def __len__(self):
        return len(self._image_states)

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self._root, filename))

    def get_sound_normalization(self):
        return self._sound_normalization




class DCAMultiAtariDataModule(LightningDataModule):
    def __init__(self, dataset, data_dir, scenario_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = str.lower(dataset)
        self.data_dir = data_dir
        self.data_config = data_config
        self.scenario_config = scenario_config

        # Data-specific variables - fill with setup function;
        self.train_data, self.val_data, self.test_data = None, None, None
        self.test_sampler = None
        self.dca_partial_eval_indices = None

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))


    def set_dca_eval_sample_indices(self):
        if self.dca_partial_eval_indices is None:
            self.dca_partial_eval_indices = np.random.choice(
                list(range(len(self.test_data))),
                self.data_config["n_dca_samples"],
                replace=False,
            )

    def prepare_data(self):
        # generate if it does not exist
        train_dataset_filename, test_dataset_filename = generate_dataset_filename(scenario=self.dataset, scenario_cfg=self.scenario_config)
        if not self._check_exists(train_dataset_filename):

            print('[MultiAtari] Dataset not found. Generate it!')


    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            atari_full = AtariDataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config, train=True)
            train_partition = int(len(atari_full)*0.9)
            val_partition = len(atari_full) - train_partition
            self.train_data, self.val_data = random_split(atari_full, [train_partition, val_partition])
            self.dims = tuple(self.train_data[0][0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = AtariDataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.test_data[0][0][0].shape)
                
            self.set_dca_eval_sample_indices()
            self.partial_test_sampler = torch.utils.data.SubsetRandomSampler(
                self.dca_partial_eval_indices
            )


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