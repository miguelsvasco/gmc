import torch
import torch.optim as optim
from collections import OrderedDict
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ModelLearner(LightningModule):
    def __init__(
        self, model, scenario, train_config, scenario_config, experiment_config,
    ):
        super(ModelLearner, self).__init__()

        self.model = model
        self.scenario = scenario
        self.experiment_config = experiment_config
        self.train_config = train_config
        self.scenario_config = scenario_config

    # ---
    # --- Pytorch ligthning --- #
    def configure_optimizers(self):
        optimiser = optim.Adam(self.model.parameters(), lr=self.train_config["learning_rate"])
        return {
        "optimizer": optimiser,
        "lr_scheduler": {
            "scheduler": ReduceLROnPlateau(optimiser, mode='min', patience=20, factor=0.1, verbose=True),
            "monitor": "val_loss",
            "frequency": 1
        },
    }

    def training_step(self, batch, batch_idx):
        # Forward pass through the encoders
        batch_X, batch_Y, batch_META = batch[0], batch[1], batch[2]

        sample_ind, text, audio, vision = batch_X
        target_data = batch_Y.squeeze(-1)  # if num of labels is 1
        data = [text, audio, vision]

        # Forward
        loss, tqdm_dict = self.model.training_step(data, target_data, self.train_config)

        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )

        return output

    def training_epoch_end(self, outputs):
        log_keys = list(outputs[0]["log"].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx]["log"][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
                .item()
            )

            # Add to sacred
            self.logger.log_metric(
                f"train_{log_key}", avg_batch_log, self.current_epoch
            )

    def validation_step(self, batch, batch_idx):

        # Forward pass through the encoders
        batch_X, batch_Y, batch_META = batch[0], batch[1], batch[2]

        sample_ind, text, audio, vision = batch_X
        target_data = batch_Y.squeeze(-1)  # if num of labels is 1
        data = [text, audio, vision]

        output_dict = self.model.validation_step(data, target_data, self.train_config)

        return output_dict

    def validation_epoch_end(self, outputs):
        log_keys = list(outputs[0].keys())
        for log_key in log_keys:
            avg_batch_log = (
                torch.stack(
                    [
                        outputs[batch_output_idx][log_key]
                        for batch_output_idx in range(len(outputs))
                    ]
                )
                .mean()
                .item()
            )
            self.log(f"val_{log_key}", avg_batch_log, on_epoch=True, logger=False, batch_size=24)
            self.logger.log_metric(f"val_{log_key}", avg_batch_log, self.current_epoch)

