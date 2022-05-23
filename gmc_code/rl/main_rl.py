import os
import torch
import sacred
import gmc_code.rl.ingredients.exp_ingredients as sacred_exp
import gmc_code.rl.ingredients.machine_ingredients as sacred_machine
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from gmc_code.rl.modules.trainers.model_trainer import ModelLearner
from gmc_code.rl.modules.sacred_loggers import SacredLogger
from gmc_code.rl.data_modules.envs.atari_env import AtariEnv
from gmc_code.rl.modules.callbacks import (
    OnEndModelTrainingPendulum,
    OnEndControllerTrainingPendulumPostEvalCb,
    OnControllerEvalPendulumPostEvalCb,
)
from gmc_code.rl.utils.general_utils import (
    setup_dca_evaluation_trainer,
    setup_model,
    setup_data_module,
    load_model,
    setup_downstream_controller,
    setup_downstream_controller_trainer,
    load_down_controller,
)

AVAIL_GPUS = min(1, torch.cuda.device_count())

ex = sacred.Experiment(
    "GMC_experiment_rl",
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

    # Init Trainer
    model_trainer = ModelLearner(
        model=model,
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
    )

    # Init Data Module
    data_module = setup_data_module(
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
    if exp_cfg["scenario"] == "pendulum":
        end_callback = OnEndModelTrainingPendulum()

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
    )

    # Train
    trainer.fit(model_trainer, data_module)


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
        scenario=exp_cfg["scenario"],
        machine_path=_config["machine"]["m_path"],
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
def train_downstream_controller(_config, _run):

    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["experiment"]["down_train_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    rep_model = load_model(sacred_config=_config, model_file=model_file)

    # Init Data Module
    rl_env = AtariEnv(
        scenario=exp_cfg["scenario"], scenario_cfg=scenario_cfg, seed=exp_cfg["seed"]
    )

    # Init downstream model
    controller = setup_downstream_controller(
        scenario=exp_cfg["scenario"],
        model_config=model_cfg,
        n_actions=rl_env.get_n_actions(),
        layer_sizes=down_train_cfg["layer_sizes"],
    )
    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Callbacks
    checkpoint_dir = log_dir_path("rl_checkpoints")

    if exp_cfg["scenario"] == "pendulum":
        post_cb = OnEndControllerTrainingPendulumPostEvalCb(
            model=model_cfg["model"],
            controller=controller,
            checkpoint_dir=checkpoint_dir,
            logger=sacred_logger,
        )

    else:
        raise ValueError("Error scenario")

    # Init Trainer
    rl_trainer = setup_downstream_controller_trainer(
        scenario=exp_cfg["scenario"],
        model=rep_model,
        controller=controller,
        env=rl_env,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
        logger=sacred_logger,
        modalities=down_train_cfg["training_mods"],
    )

    # Train
    rl_trainer.train_controller(post_cb=post_cb)
    rl_env.close()


@ex.capture
def eval_downstream_controller(_config, _run):

    # Create folders to hold results
    if (
        0 in _config["experiment"]["evaluation_mods"]
        and 1 in _config["experiment"]["evaluation_mods"]
    ):
        checkpoint_dir = log_dir_path("results_rl/joint")
        os.makedirs(log_dir_path("results_rl/joint"), exist_ok=True)
    elif 0 in _config["experiment"]["evaluation_mods"]:
        checkpoint_dir = log_dir_path("results_rl/image")
        os.makedirs(log_dir_path("results_rl/image"), exist_ok=True)
    else:
        checkpoint_dir = log_dir_path("results_rl/sound")
        os.makedirs(log_dir_path("results_rl/sound"), exist_ok=True)

    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    down_train_cfg = _config["experiment"]["down_train_config"]
    down_eval_cfg = _config["experiment"]["down_eval_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Load model
    model_file = trained_model_dir_path(exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    rep_model = load_model(sacred_config=_config, model_file=model_file)

    # Init Data Module
    rl_env = AtariEnv(
        scenario=exp_cfg["scenario"], scenario_cfg=scenario_cfg, seed=exp_cfg["seed"]
    )

    # Init downstream model
    controller_file = trained_model_dir_path("down_" + exp_cfg['model'] + "_" + exp_cfg['scenario'] + "_model.pth.tar")
    controller = load_down_controller(
        sacred_config=_config,
        n_actions=rl_env.get_n_actions(),
        layer_sizes=down_train_cfg["layer_sizes"],
        down_model_file=controller_file
    )
    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Callbacks
    if exp_cfg["scenario"] == "pendulum":
        post_cb = OnControllerEvalPendulumPostEvalCb(
            model=model_cfg["model"],
            controller=controller,
            checkpoint_dir=checkpoint_dir,
            logger=sacred_logger,
        )
    else:
        raise ValueError("Error Scenario")

    # Init Trainer
    rl_trainer = setup_downstream_controller_trainer(
        scenario=exp_cfg["scenario"],
        model=rep_model,
        controller=controller,
        env=rl_env,
        scenario_config=scenario_cfg,
        train_config=down_train_cfg,
        logger=sacred_logger,
        modalities=exp_cfg["evaluation_mods"],
    )

    # Train
    if exp_cfg["scenario"] == "pendulum":
        rl_trainer.evaluate_controller(
            max_episodes=down_eval_cfg["max_evaluation_episodes"],
            max_episode_length=down_eval_cfg["max_evaluation_episode_length"],
            post_episode_cb=post_cb,
        )
    else:
        raise ValueError("Error Scenario")

    rl_env.close()


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

    elif _config["experiment"]["stage"] == "train_downstream_controller":
        os.makedirs(log_dir_path("rl_checkpoints"), exist_ok=True)
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_downstream_controller()

    elif _config["experiment"]["stage"] == "evaluate_downstream_controller":
        eval_downstream_controller()

    else:
        raise ValueError(
            "[RL Experiment] Incorrect stage of pipeline selected: "
            + str(_config["experiment"]["stage"])
        )


if __name__ == "__main__":
    ex.run_commandline()

