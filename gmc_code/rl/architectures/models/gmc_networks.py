import torch
import torch.nn as nn
import torch.nn.functional as F

# Pendulum
class PendulumCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(PendulumCommonEncoder, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 128), Swish(), nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class PendulumImageProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumImageProcessor, self).__init__()
        self.common_dim = common_dim

        self.image_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish(),
        )

        self.projector = nn.Linear(14400, common_dim)

    def forward(self, x):
        x = self.image_features(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)


class PendulumSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumSoundProcessor, self).__init__()

        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
            self.n_stack * self.sound_channels * self.sound_length
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
        )

        self.projector = nn.Linear(50, common_dim)

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        return self.projector(h)


# Pendulum
class PendulumJointProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumJointProcessor, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
            self.n_stack * self.sound_channels * self.sound_length
        )

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish(),
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
        )

        self.projector = nn.Linear(14400 + 50, common_dim)

    def forward(self, x):

        x_img, x_snd = x[0], x[1]

        x_img = self.img_features(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        return self.projector(torch.cat((x_img, x_snd), dim=-1))

"""


Extra components


"""

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)