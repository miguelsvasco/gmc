import torch
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
from gmc_code.rl.data_modules.utils.game_utils import pendulum_image_preprocess, pendulum_sound_preprocess


class Processor(LightningModule):
    def __init__(self, mods):
        super(Processor, self).__init__()
        self.mods = mods
        self.n_mods = len(mods)

    def get_n_mods(self):
        return self.n_mods
    def get_mods(self):
        return self.mods
    def preprocess(self, observation):
        return
    def postprocess(self, observation):
        return


class PendulumProcessor(Processor):
    def __init__(self, mods):
        super(PendulumProcessor, self).__init__(mods)
        self.sound_norm = None

    def set_sound_norm(self, sound_norm):
        self.sound_norm = sound_norm

    def get_sound_norm(self):
        return self.sound_norm

    def preprocess(self, observation):

        if self.sound_norm is None:
            raise ValueError('[Pendulum Processor] No sound normalization factor available. Please set it first.')

        image, sound = observation

        # Preprocess image the same way as the dataset
        image_p = pendulum_image_preprocess(image)

        # Preprocess sound
        sound_p = pendulum_sound_preprocess(sound, self.sound_norm)

        # Torch it up
        image_p = torch.tensor(image_p).float()
        sound_p = torch.tensor(np.array(sound_p)).float()

        # Fix sizes
        image_p = image_p.unsqueeze(0).unsqueeze(0)
        sound_p = sound_p.unsqueeze(0).unsqueeze(0)

        # Send it up
        pre_obs = []
        if 0 in self.mods:
            pre_obs.append(image_p)
        if 1 in self.mods:
            pre_obs.append(sound_p)

        return pre_obs


    def eval_preprocess(self, observation):

        if self.sound_norm is None:
            raise ValueError('[Pendulum Processor] No sound normalization factor available. Please set it first.')

        image, sound = observation

        # Preprocess image the same way as the dataset
        image_p = pendulum_image_preprocess(image)

        # Preprocess sound
        sound_p = pendulum_sound_preprocess(sound, self.sound_norm)

        # Torch it up
        image_p = torch.tensor(image_p).float()
        sound_p = torch.tensor(np.array(sound_p)).float()

        # Fix sizes
        image_p = image_p.unsqueeze(0).unsqueeze(0)
        sound_p = sound_p.unsqueeze(0).unsqueeze(0)

        # Send it up
        pre_obs = []
        if 0 in self.mods:
            pre_obs.append(image_p)
        if 1 in self.mods:
            pre_obs.append(sound_p)

        return pre_obs


    def postprocess(self, observation):

        if 0 in self.mods and 1 in self.mods:
            image, sound = observation
        elif 0 in self.mods:
            image = observation[0]
            sound = None
        elif 1 in self.mods:
            image = None
            sound = observation[0]
        else:
            raise ValueError("[Processor] No modalities selected for Pendulum scenario.")

        if image is not None:
            image = torch.cat(image, dim=1)
        if sound is not None:
            sound = torch.cat(sound, dim=1)

        with torch.no_grad():
            return [image, sound]