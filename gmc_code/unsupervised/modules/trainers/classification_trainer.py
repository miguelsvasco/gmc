import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from pytorch_lightning import LightningModule


class ClassifierLearner(LightningModule):
    def __init__(self, model, scenario, classifier, train_config, modalities=None):
        super(ClassifierLearner, self).__init__()

        self.scenario = scenario
        self.model = model
        self.model.eval()
        self.classifier = classifier
        self.train_config = train_config
        self.criterion = nn.CrossEntropyLoss()
        self.test_modalities = modalities

        # initialize metric
        if self.scenario == 'mhd':
            self.accuracy_metric = torchmetrics.Accuracy(num_classes=10)
        else:
            raise ValueError("[Classifier Learner] Scenario not implement: " + str(self.scenario))

    # ---
    # --- Pytorch ligthning --- #
    def configure_optimizers(self):
        optimiser = optim.Adam(
            self.classifier.parameters(), lr=self.train_config["learning_rate"]
        )
        return optimiser

    def training_step(self, batch, batch_idx):
        # Forward pass through the encoders
        if self.scenario == 'mhd':
            data = [batch[0], batch[1], batch[2], torch.nn.functional.one_hot(batch[3], num_classes=10).float()]
            targets = batch[3]
        else:
            raise ValueError(
                "[Downstream Classifier Learner] Scenario not yet implemented: " + str(self.scenario)
            )

        latent = self.model.encode(data)
        logits = self.classifier(latent)
        loss = self.criterion(logits, targets)

        tqdm_dict = {"loss": loss}
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

        if self.scenario == 'mhd':
            data = [batch[0], batch[1], batch[2], torch.nn.functional.one_hot(batch[3], num_classes=10).float()]
            targets = batch[3]
        else:
            raise ValueError(
                "[Downstream Classifier Learner] Scenario not yet implemented: " + str(self.scenario)
            )

        latent = self.model.encode(data)
        logits = self.classifier(latent)
        loss = self.criterion(logits, targets)

        output_dict = {"loss": loss}
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

    def test_step(self, batch, batch_idx):

        if self.scenario == 'mhd':
            data = [batch[0], batch[1], batch[2], torch.nn.functional.one_hot(batch[3], num_classes=10).float()]
            targets = batch[3]
        else:
            raise ValueError(
                "[Downstream Classifier Learner] Scenario not yet implemented: " + str(self.scenario)
            )

        # Drop modalities (if required)
        input_data = []
        if self.test_modalities is not None:
            for j in range(len(data)):
                if j not in self.test_modalities:
                    input_data.append(None)
                else:
                    input_data.append(data[j])
        else:
            input_data = data

        latent = self.model.encode(input_data)
        logits = self.classifier(latent)

        loss = self.criterion(logits, targets)
        acc = self.accuracy_metric(logits, targets)

        output_dict = {"loss": loss, "acc": acc}
        return output_dict

    def test_epoch_end(self, outputs):
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
            self.log(f"test_{log_key}", avg_batch_log, logger=False)
            self.logger.log_metric(f"test_{log_key}", avg_batch_log)

