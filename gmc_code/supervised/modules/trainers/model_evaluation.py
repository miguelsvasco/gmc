from tqdm import tqdm
from pytorch_lightning import LightningModule
from gmc_code.supervised.modules.trainers.model_evaluation_metrics import *



class ModelEvaluation(LightningModule):
    def __init__(self, model_name, model, scenario, test_loader, sacred_logger, modalities=None):
        super(ModelEvaluation, self).__init__()

        self.scenario = scenario
        self.model_name = model_name

        self.model = model
        self.model.eval()

        self.test_modalities = modalities
        self.test_loader = test_loader
        self.sacred_logger = sacred_logger

    def evaluate(self):

        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in tqdm(enumerate(self.test_loader)):

                sample_ind, text, audio, vision = batch_X
                data = [text, audio, vision]
                target_data = batch_Y.squeeze(-1)  # if num of labels is 1

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

                # Parallel model
                preds = self.model.encode(input_data)

                if self.scenario == 'iemocap':
                    preds = preds.view(-1, 2)
                    target_data = target_data.view(-1)

                # Collect the results into dictionary
                results.append(preds)
                truths.append(target_data)

            results = torch.cat(results)
            truths = torch.cat(truths)

            if self.scenario == "mosei":
                eval_mosei(results, truths, self.sacred_logger, True)
            elif self.scenario == 'mosi':
                eval_mosi(results, truths, self.sacred_logger, True)




