from gmc_code.rl.architectures.models.gmc import PendulumGMC
from gmc_code.rl.architectures.downstream.ddpg import DDPG
from gmc_code.rl.modules.trainers.controller_trainer import DDPGLearner
from gmc_code.rl.modules.trainers.dca_evaluation_trainer import DCAEvaluator
from gmc_code.rl.data_modules.DCAMultiAtari_dataset import *
from gmc_code.rl.data_modules.MultiAtari_dataset import MultiAtariDataModule


"""

Setup functions

"""


def setup_model(scenario, model, model_config, scenario_config):
    if model == "gmc":
        if scenario == "pendulum":
            return PendulumGMC(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
            )

        else:
            raise ValueError(
                "[Model Setup] Selected scenario not yet implemented for GMC Single model: "
                + str(scenario)
            )

    else:
        raise ValueError(
            "[Model Setup] Selected model not yet implemented: " + str(model)
        )


def setup_data_module(scenario, experiment_config, scenario_config, train_config):
    if experiment_config["stage"] == "evaluate_dca":
        return DCAMultiAtariDataModule(
            dataset=scenario,
            scenario_config=scenario_config,
            data_dir=scenario_config["data_dir"],
            data_config=train_config,
        )

    elif scenario == "pendulum":
        return MultiAtariDataModule(
            dataset=scenario,
            data_dir=scenario_config["data_dir"],
            scenario_config=scenario_config,
            data_config=train_config,
        )
    else:
        raise ValueError(
            "[Data Module Setup] Selected Module not yet implemented: " + str(scenario)
        )


def setup_dca_evaluation_trainer(model, machine_path, scenario, config):
    return DCAEvaluator(
        model=model,
        scenario=scenario,
        machine_path=machine_path,
        minimum_cluster_size=config["minimum_cluster_size"],
        unique_modality_idxs=config["unique_modality_idxs"],
        unique_modality_dims=config["unique_modality_dims"],
        partial_modalities_idxs=config["partial_modalities_idxs"],
    )


def setup_downstream_controller(scenario, model_config, n_actions, layer_sizes):
    if scenario == "pendulum":
        return DDPG(
            model_config["latent_dim"], n_actions=n_actions, layer_sizes=layer_sizes
        )
    else:
        raise ValueError(
            "[Down Controller Setup] Selected scenario not yet implemented: "
            + str(scenario)
        )


def setup_downstream_controller_trainer(
    scenario, model, controller, env, scenario_config, train_config, logger, modalities
):
    if scenario == "pendulum":  # Classification
        return DDPGLearner(
            model=model,
            controller=controller,
            env=env,
            scenario_config=scenario_config,
            train_config=train_config,
            logger=logger,
            modalities=modalities,
        )
    else:
        raise ValueError(
            "[Down Controller Trainer Setup] Trainer for selected scenario not yet implemented: "
            + str(scenario)
        )


"""

Loading functions

"""


def load_model(sacred_config, model_file):

    model = setup_model(
        scenario=sacred_config["experiment"]["scenario"],
        model=sacred_config["experiment"]["model"],
        model_config=sacred_config["experiment"]["model_config"],
        scenario_config=sacred_config["experiment"]["scenario_config"],
    )

    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    model.freeze()

    return model


def load_down_controller(sacred_config, down_model_file, n_actions, layer_sizes):

    down_model = setup_downstream_controller(
        scenario=sacred_config["experiment"]["scenario"],
        model_config=sacred_config["experiment"]["model_config"],
        n_actions=n_actions,
        layer_sizes=layer_sizes,
    )

    checkpoint = torch.load(down_model_file)
    down_model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    down_model.freeze()

    return down_model


"""


General functions

"""


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )

