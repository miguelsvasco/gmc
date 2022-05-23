import torch
from pytorch_lightning import LightningModule
from collections import OrderedDict
import torch.optim as optim


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
        optimiser = optim.Adam(
            self.model.parameters(), lr=self.train_config["learning_rate"]
        )
        return optimiser

    def training_step(self, batch, batch_idx):
        # Forward pass through the encoders
        data = batch[0]
        loss, tqdm_dict = self.model.training_step(data, self.train_config)
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

        data = batch[0]
        output_dict = self.model.validation_step(data, self.train_config)
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
            self.log(f"val_{log_key}", avg_batch_log, on_epoch=True, logger=False)
            self.logger.log_metric(f"val_{log_key}", avg_batch_log, self.current_epoch)

