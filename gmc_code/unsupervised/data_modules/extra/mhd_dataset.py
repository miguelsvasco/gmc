import os
import torch
from torch.utils.data import Dataset

def unstack_tensor(tensor, dim=0):

    tensor_lst = []
    for i in range(tensor.size(dim)):
        tensor_lst.append(tensor[i])
    tensor_unstack = torch.cat(tensor_lst, dim=0)
    return tensor_unstack

class MHDDataset(Dataset):
    def __init__(self, data_file, train=True):


        self.train = train
        if train:
            self.data_file = os.path.join(data_file, "mhd_train.pt")
        else:
            self.data_file = os.path.join(data_file, "mhd_test.pt")

        if not os.path.exists(data_file):
                raise RuntimeError(
                    'MHD Dataset not found. Please generate dataset and place it in the data folder.')

        # Load data
        import ipdb; ipdb.set_trace()
        self._label_data, self._image_data, self._trajectory_data, self._sound_data, self._traj_normalization, self._sound_normalization = torch.load(self.data_file)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            array: image_data, sound_data, trajectory_data, label_data
        """
        audio = unstack_tensor(self._sound_data[index]).unsqueeze(0)
        audio_perm = audio.permute(0, 2, 1)
        import ipdb; ipdb.set_trace()
        return self._image_data[index], audio_perm, self._trajectory_data[index], self._label_data[index]


    def __len__(self):
        return len(self._label_data)

    def get_audio_normalization(self):
        return self._audio_normalization

    def get_traj_normalization(self):
        return self._traj_normalization