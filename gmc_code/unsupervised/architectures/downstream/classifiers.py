from torch import nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule


class ClassifierMNIST(LightningModule):
    def __init__(self, latent_dim):
        super().__init__()

        self.layer_1 = nn.Linear(latent_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        return x