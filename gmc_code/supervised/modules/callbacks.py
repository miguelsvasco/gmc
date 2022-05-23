import os
import torch
from pytorch_lightning.callbacks import Callback

class OnEndModelTrainingMosei(Callback):

    def __init__(self, model_path, model_filename):
        super(OnEndModelTrainingMosei, self).__init__()
        self.best_model_path = model_path
        self.best_model_filename = model_filename

    def on_init_end(self, trainer):
        print(f"Initialised Model Trainer with {trainer.default_root_dir}")

    def on_train_end(self, trainer, pl_module):

        torch.save(
            {"state_dict": pl_module.model.state_dict()},
            os.path.join(
                trainer.default_root_dir, f"{pl_module.model.name}_mosei_model.pth.tar"
            ),
        )

        print(
            f"Model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the MOSEI dataset saved to {trainer.default_root_dir}"
        )


class OnEndModelTrainingMosi(Callback):

    def __init__(self, model_path, model_filename):
        super(OnEndModelTrainingMosi, self).__init__()
        self.best_model_path = model_path
        self.best_model_filename = model_filename

    def on_init_end(self, trainer):
        print(f"Initialised Model Trainer with {trainer.default_root_dir}")

    def on_train_end(self, trainer, pl_module):

        torch.save(
            {"state_dict": pl_module.model.state_dict()},
            os.path.join(
                trainer.default_root_dir, f"{pl_module.model.name}_mosi_model.pth.tar"
            ),
        )

        print(
            f"Model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the MOSI dataset saved to {trainer.default_root_dir}"
        )