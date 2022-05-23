import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset, IterableDataset
from gmc_code.rl.data_modules.utils.atari_dataset_utils import generate_dataset_filename, generate_dataset

# 'state', 'action', 'next_state', 'reward', 'terminal'

class MultiAtariDataModule(LightningDataModule):

    def __init__(self, dataset, data_dir, scenario_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = str.lower(dataset)
        self.data_dir = data_dir
        self.data_config = data_config
        self.scenario_config = scenario_config

        # Data-specific variables - fill with setup function;
        self.train_data, self.val_data, self.test_data = None, None, None

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))

    def prepare_data(self):
        # generate if it does not exist
        train_dataset_filename, test_dataset_filename = generate_dataset_filename(scenario=self.dataset, scenario_cfg=self.scenario_config)
        if not self._check_exists(train_dataset_filename):

            print('[MultiAtari] Dataset not found. Generating dataset...')
            generate_dataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config)

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



    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])


class MultiAtariDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, next_states, rewards, terminals = self.buffer.sample(self.sample_size)
        for i in range(len(terminals)):
            yield states[i], actions[i], next_states[i], rewards[i], terminals[i]


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
