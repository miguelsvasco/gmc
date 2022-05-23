import os
import torch
from pytorch_lightning.callbacks import Callback

class OnEndModelTrainingPendulum(Callback):
    def on_init_end(self, trainer):
        print(f"Initialised Model Trainer with {trainer.default_root_dir}")

    def on_train_end(self, trainer, pl_module):

        torch.save(
            {"state_dict": pl_module.model.state_dict()},
            os.path.join(
                trainer.default_root_dir,
                f"{pl_module.model.name}_pendulum_model.pth.tar",
            ),
        )

        # Send model to Sacred
        trainer.logger.log_artifact(
            name=f"{pl_module.model.name}_pendulum_model.pth.tar",
            filepath=os.path.join(
                trainer.default_root_dir,
                f"{pl_module.model.name}_pendulum_model.pth.tar",
            ),
        )

        print(
            f"Model {pl_module.model.name} trained for {trainer.max_epochs} epochs in the Pendulum dataset saved to {trainer.default_root_dir}"
        )


# Train Downstream


class OnEndControllerTrainingPendulumPostEvalCb(object):
    def __init__(self, model, controller, checkpoint_dir, logger):
        self.model = model
        self.controller = controller
        self.best_reward = float("-inf")
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger

    def record_eval_checkpoint(self, controller, info):
        avg_reward = info["eval_avg_reward"]
        frame_number = info["frame_number"]

        is_best = avg_reward > self.best_reward
        self.best_reward = max(avg_reward, self.best_reward)

        torch.save(
            {"state_dict": controller.state_dict()},
            os.path.join(
                self.checkpoint_dir,
                f"down_{self.model}_pendulum_model_check_{frame_number}.pth.tar",
            ),
        )

        if is_best:
            torch.save(
                {"state_dict": controller.state_dict()},
                os.path.join(
                    self.checkpoint_dir, f"down_{self.model}_pendulum_model.pth.tar"
                ),
            )

        # Save RL gif
        self.logger.log_rl_gif(
            observations=info["eval_observations"],
            filepath=os.path.join(
                self.checkpoint_dir, f"{self.model}_pendulum_rl_agent.gif"
            ),
            name=f"{self.model}_pendulum_rl_agent_frame_{frame_number}.gif",
        )

    def on_train_end(self, controller, info):

        avg_reward = info["eval_avg_reward"]
        is_best = avg_reward > self.best_reward
        self.best_reward = max(avg_reward, self.best_reward)

        torch.save(
            {"state_dict": controller.state_dict()},
            os.path.join(
                self.checkpoint_dir, f"down_{self.model}_pendulum_model_final.pth.tar"
            ),
        )

        if is_best:
            torch.save(
                {"state_dict": controller.state_dict()},
                os.path.join(
                    self.checkpoint_dir, f"down_{self.model}_pendulum_model.pth.tar"
                ),
            )

        # Send model to Sacred
        self.logger.log_artifact(
            name=f"down_{self.model}_pendulum_model.pth.tar",
            filepath=os.path.join(
                self.checkpoint_dir, f"down_{self.model}_pendulum_model.pth.tar"
            ),
        )

        print(
            f"Controller Model {self.model} in the Pendulum dataset saved to {self.checkpoint_dir}"
        )


class OnControllerEvalPendulumPostEvalCb(object):
    def __init__(self, model, controller, checkpoint_dir, logger):
        self.model = model
        self.controller = controller
        self.checkpoint_dir = checkpoint_dir
        self.logger = logger

    def on_eval_episode_end(self, info):

        # Save RL gif
        self.logger.log_rl_eval_gif(
            observations=info["eval_observations"],
            filepath=os.path.join(
                self.checkpoint_dir,
                f'eval_{self.model}_pendulum_rl_episode_{info["episode_number"]}.gif',
            ),
            name=f'eval_{self.model}_pendulum_rl_agent_episode_{info["episode_number"]}.gif',
        )

