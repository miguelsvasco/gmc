import os
import torch
import sacred
import gmc_code.supervised.ingredients.exp_ingredients as sacred_exp
import gmc_code.supervised.ingredients.machine_ingredients as sacred_machine
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from gmc_code.supervised.modules.trainers.model_trainer import ModelLearner
from gmc_code.supervised.modules.trainers.model_evaluation import ModelEvaluation
from gmc_code.supervised.modules.callbacks import OnEndModelTrainingMosi, OnEndModelTrainingMosei
from gmc_code.supervised.modules.sacred_loggers import SacredLogger

from gmc_code.supervised.utils.general_utils import (
    setup_model,
    setup_data_module,
    load_model,
    setup_dca_evaluation_trainer,
)

AVAIL_GPUS = min(1, torch.cuda.device_count())

ex = sacred.Experiment(
    "GMC_experiment_supervised",
    ingredients=[sacred_machine.machine_ingredient, sacred_exp.exp_ingredient],
)


@ex.capture
def log_dir_path(folder, _config, _run):

    model_type = str(_config["experiment"]["model"])
    exp_name = str(_config["experiment"]["scenario"])

    return os.path.join(
        _config["machine"]["m_path"],
        "evaluation/",
        model_type + "_" + exp_name,
        f'log_{_config["experiment"]["seed"]}',
        folder,
    )

@ex.capture
def trained_model_dir_path(file, _config, _run):
    return os.path.join(
        _config["machine"]["m_path"],
        "trained_models/",
        file
    )



@ex.capture
def load_hyperparameters(_config, _run):
    exp_cfg = _config["experiment"]
    scenario_cfg = _config["experiment"]["scenario_config"]
    model_cfg = _config["experiment"]["model_config"]

    return exp_cfg, scenario_cfg, model_cfg


@ex.capture
def train_model(_config, _run):

    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    model_train_cfg = _config["experiment"]["model_train_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Init model
    model = setup_model(
        model=exp_cfg["model"],
        scenario=exp_cfg["scenario"],
        scenario_config=scenario_cfg,
        model_config=model_cfg,
    )

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
    )

    # Init Trainer
    model_trainer = ModelLearner(
        model=model,
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Train
    checkpoint_dir = log_dir_path("checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}-"
        + "{epoch:02d}",
        monitor="val_loss",
        every_n_epochs=model_train_cfg["snapshot"],
        save_top_k=-1,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Callbacks
    if exp_cfg["scenario"] == "mosei":
        end_callback = OnEndModelTrainingMosei(model_path=log_dir_path("checkpoints"),
                                               model_filename=f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last.pth")

    elif exp_cfg["scenario"] == "mosi":
        end_callback = OnEndModelTrainingMosi(model_path=log_dir_path("checkpoints"),
                                              model_filename=f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last.pth")

    else:
        raise ValueError("Error")

    # TEST_limit_train_batches = 0.01
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=model_train_cfg["epochs"],
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("saved_models"),
        logger=sacred_logger,
        callbacks=[checkpoint_callback, end_callback],
        gradient_clip_val=0.8,
    )

    # Train
    trainer.fit(model_trainer, data_module)

    sacred_logger.log_artifact(
        name=f"{exp_cfg['model']}_{exp_cfg['scenario']}_model.pth.tar",
        filepath=os.path.join(log_dir_path("checkpoints"),
                              f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last.pth"))


@ex.capture
def dca_eval_model(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, _ = load_hyperparameters()
    dca_eval_cfg = _config["experiment"]["dca_evaluation_config"]

    # Set the seeds
    seed_everything(dca_eval_cfg["random_seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    model = load_model(sacred_config=_config, model_file=model_file)

    # Init Trainer
    dca_trainer = setup_dca_evaluation_trainer(
        model=model,
        machine_path=_config["machine"]["m_path"],
        scenario=exp_cfg["scenario"],
        config=dca_eval_cfg,
    )

    # Init Data Module
    dca_data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=dca_eval_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=10,
        default_root_dir=log_dir_path("results_dca_evaluation"),
        logger=sacred_logger,
    )

    trainer.test(dca_trainer, dca_data_module)
    return

@ex.capture
def evaluate(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["experiment"]["down_train_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    encoder_model = load_model(sacred_config=_config, model_file=model_file)

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
    )
    data_module.setup(stage='test')
    test_dataloader = data_module.test_dataloader()

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Evaluator
    affect_evaluator = ModelEvaluation(model_name=exp_cfg['model'],
                                        model=encoder_model,
                                        scenario=exp_cfg['scenario'],
                                        sacred_logger=sacred_logger,
                                        test_loader=test_dataloader,
                                        modalities=_config["experiment"]["evaluation_mods"])

    affect_evaluator.evaluate()


@ex.main
def main(_config, _run):

    # Run experiment
    if _config["experiment"]["stage"] == "train_model":
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_model()

    elif _config["experiment"]["stage"] == "evaluate_dca":
        os.makedirs(log_dir_path("results_dca_evaluation"), exist_ok=True)
        dca_eval_model()

    elif _config["experiment"]["stage"] == "evaluate_downstream_classifier":
        os.makedirs(log_dir_path("results_down"), exist_ok=True)
        evaluate()

    else:
        raise ValueError(
            "[Supervised Experiment] Incorrect stage of pipeline selected: "
            + str(_config["experiment"]["stage"])
        )


if __name__ == "__main__":
    ex.run_commandline()

