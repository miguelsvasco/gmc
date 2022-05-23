import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from gym.wrappers.time_limit import TimeLimit as TimeLimitWrapper
from gmc_code.rl.utils.rl_utils import *
from gmc_code.rl.architectures.downstream.buffers import FrameBuffer
from gmc_code.rl.data_modules.MultiAtari_dataset import AtariDataset
from gmc_code.rl.architectures.downstream.processor import PendulumProcessor
from gmc_code.rl.architectures.downstream.buffers import PendulumReplayMemory
from gmc_code.rl.architectures.downstream.policy import PendulumPolicy


class ControllerLearner(LightningModule):
    def __init__(
        self,
        model,
        controller,
        env,
        scenario_config,
        train_config,
        logger,
        modalities=None,
    ):

        super(ControllerLearner, self).__init__()

        self.model = model
        self.model.eval()
        self.controller = controller
        self.train_params = train_config
        self.modalities = modalities
        self.sacred_logger = logger
        self.post_cb = None

        # Setup Environment and RL stuff
        self.env = self.setup_env(train_config=train_config, env=env)
        self.processor = self.setup_processor(train_config, scenario_config)
        self.memory = self.setup_memory(train_config, scenario_config)
        self.policy = self.setup_policy(
            self.env, self.controller, train_config, scenario_config
        )
        self.frame_buffer = FrameBuffer(
            frames_per_state=scenario_config["n_stack"], processor=self.processor
        )

        # Setup training variables
        self.gamma = train_config["gamma"]
        self.batch_size = train_config["batch_size"]
        self.max_frames = train_config["max_frames"]
        self.eval_frequency = train_config["eval_frequency"]
        self.eval_length = train_config["eval_length"]
        self.memory_size = train_config["memory_size"]

    # Reset environment
    def reset_env(self):
        self.env.reset()

    # Setup Environment
    def setup_env(self, train_config, env):
        if train_config["max_episode_length"] > 0:
            env = TimeLimitWrapper(env, train_config["max_episode_length"])
        return env

    # Setup Processor
    def setup_processor(self, train_config, scenario_config):
        if scenario_config["scenario"] == "pendulum":
            pend_processor = PendulumProcessor(mods=self.modalities)

            # Load Sound Normalization from dataset
            pend_dataset = AtariDataset(
                scenario="pendulum",
                scenario_cfg=scenario_config,
                data_dir=scenario_config["data_dir"],
            )

            pend_processor.set_sound_norm(pend_dataset.get_sound_normalization())

            return pend_processor
        else:
            raise ValueError(
                "[Setup Processor] Scenario not yet implemented: "
                + str(scenario_config["scenario"])
            )

    # Setup Memory
    def setup_memory(self, train_config, scenario_config):
        if scenario_config["scenario"] == "pendulum":
            return PendulumReplayMemory(capacity=train_config["memory_size"])
        else:
            raise ValueError(
                "[Setup Memory] Scenario not yet implemented: "
                + str(scenario_config["scenario"])
            )

    # Setup Memory
    def setup_policy(self, env, model, train_config, scenario_config):
        if scenario_config["scenario"] == "pendulum":
            return PendulumPolicy(
                policy_net=model.actor,
                action_space=env.action_space,
                controller_config=train_config,
            )
        else:
            raise ValueError(
                "[Setup Policy] Scenario not yet implemented: "
                + str(scenario_config["scenario"])
            )

    def train_controller(self, post_cb):
        return

    def val_controller(self, frame_number, episode_number, n_eval_episodes):
        return


"""

DDPG Learner

"""


class DDPGLearner(ControllerLearner):
    def __init__(
        self,
        model,
        controller,
        env,
        scenario_config,
        train_config,
        logger,
        modalities=None,
    ):
        super(DDPGLearner, self).__init__(
            model, controller, env, scenario_config, train_config, logger, modalities
        )

        self.actor_optim = optim.Adam(
            self.controller.actor.parameters(), lr=train_config["actor_learning_rate"]
        )
        self.critic_optim = optim.Adam(
            self.controller.critic.parameters(), lr=train_config["critic_learning_rate"]
        )

        # Extra training variables
        self.tau = train_config["tau"]

    def train_controller(self, post_cb):

        self.post_cb = post_cb

        self.model.eval()
        self.controller.train()

        frame_number = 0
        episode_number = 0
        avg_episode_total_reward = FixedHorizonAverageMeter(50)
        avg_rewards = FixedHorizonAverageMeter(1000)
        avg_critic_losses = FixedHorizonAverageMeter(1000)
        avg_actor_losses = FixedHorizonAverageMeter(1000)

        new_episode = True
        while frame_number < self.max_frames:
            if new_episode:
                self.frame_buffer.reset()
                observation = self.env.reset()
                self.frame_buffer.append(observation)
                observation_state = self.frame_buffer.get_state()
                latent_state = self.model.encode(observation_state)

                print(
                    f"Train Episode: {episode_number} - {frame_number}/{self.max_frames}"
                )

                episode_rewards = []

                new_episode = False

            action = self.policy.select_action(latent_state, frame_number)
            next_observation, reward, done, info = self.env.step(action)

            self.frame_buffer.append(next_observation)

            next_observation_state = self.frame_buffer.get_state()
            next_latent_state = self.model.encode(next_observation_state)
            torch_action, torch_reward = (
                torch.from_numpy(action).to(self.device),
                torch.tensor([reward], device=self.device),
            )
            self.memory.push(
                latent_state, torch_action, next_latent_state, torch_reward, done
            )
            latent_state = next_latent_state

            replay_memory_filled = frame_number > self.memory_size
            if replay_memory_filled:
                critic_loss, actor_loss = self.optimize_model()
                avg_critic_losses.update(critic_loss)
                avg_actor_losses.update(actor_loss)

            # log every 500 frames
            should_log = frame_number % 500 == 0
            if should_log and replay_memory_filled:
                print(
                    f"===> Train Episode: {episode_number} - {frame_number}/{self.max_frames}\t"
                    f"Episode avg critic loss: {avg_critic_losses.avg:.3f}\t"
                    f"Episode avg actor loss: {avg_actor_losses.avg:.3f}\t"
                    f"Episode avg episode total reward: {avg_episode_total_reward.avg:.3f}\t"
                    f"Episode avg reward: {avg_rewards.avg:.3f}\t"
                    f"ReplayBuf avg reward: {self.memory.stats()[0]:.3f}"
                )
            elif should_log and (not replay_memory_filled):
                print(f"Fill ReplayMemory: {frame_number}/{self.memory_size}")

            avg_rewards.update(reward)
            episode_rewards.append(reward)
            if done:
                total_episode_reward = discount_rewards(episode_rewards, self.gamma)
                avg_episode_total_reward.update(total_episode_reward)

                self.sacred_logger.log_metric(
                    name="avg_critic_loss",
                    value=avg_critic_losses.avg,
                    step=frame_number,
                )
                self.sacred_logger.log_metric(
                    name="avg_actor_loss", value=avg_actor_losses.avg, step=frame_number
                )
                self.sacred_logger.log_metric(
                    name="avg_reward", value=avg_rewards.avg, step=frame_number
                )
                self.sacred_logger.log_metric(
                    name="avg_episode_total_reward",
                    value=avg_episode_total_reward.avg,
                    step=frame_number,
                )
                self.sacred_logger.log_metric(
                    name="last_episode_total_reward",
                    value=total_episode_reward,
                    step=frame_number,
                )
                self.sacred_logger.log_metric(
                    name="replay_buf_avg_reward",
                    value=self.memory.stats()[0],
                    step=frame_number,
                )

                episode_number += 1
                new_episode = True

                should_eval = episode_number % self.eval_frequency == 0
                if should_eval:
                    self.val_controller(frame_number, episode_number, self.eval_length)
                    self.controller.train()
                    self.model.eval()

            frame_number += 1

        # Last validation
        self.val_controller(
            frame_number, episode_number, self.eval_length, end_val=True
        )

    def val_controller(
        self, frame_number, episode_number, n_eval_episodes, end_val=False
    ):

        print(f"**** Eval Episode: {episode_number}")

        self.model.eval()
        self.controller.eval()

        observations = []
        total_rewards = []
        self.reset_env()  # force a reset
        for episode in range(n_eval_episodes):
            print(f"=====Eval epoch: {episode}/{n_eval_episodes}")
            self.frame_buffer.reset()
            observation = self.env.reset()
            self.frame_buffer.append(observation)
            observation_state = self.frame_buffer.get_state()
            latent_state = self.model.encode(observation_state).detach()

            episode_rewards = []
            episode_observations = []

            done = False
            while not done:
                action = self.policy.select_eval_action(latent_state)

                next_observation, reward, done, info = self.env.step(action)
                self.frame_buffer.append(next_observation)
                observation_state = self.frame_buffer.get_state()
                latent_state = self.model.encode(observation_state).detach()

                episode_rewards.append(reward)
                episode_observations.append(next_observation)

            total_rewards.append(discount_rewards(episode_rewards, self.gamma))
            observations.append(episode_observations)

        self.sacred_logger.log_metric(
            name="eval_avg_reward", value=np.mean(total_rewards), step=frame_number
        )

        info = {}
        info["frame_number"] = frame_number
        info["eval_avg_reward"] = np.mean(total_rewards)
        info["eval_observations"] = observations
        if not end_val:
            self.post_cb.record_eval_checkpoint(controller=self.controller, info=info)
        else:
            self.post_cb.on_train_end(controller=self.controller, info=info)

    def evaluate_controller(
        self, max_episodes, max_episode_length, post_episode_cb=None
    ):

        self.model.eval()
        self.controller.eval()

        avg_reward_list = []
        for episode_number in range(max_episodes):
            print(f"Eval Episode: {episode_number}/{max_episodes}")

            self.frame_buffer.reset()
            observation = self.env.reset()
            self.frame_buffer.append(observation)
            observation_state = self.frame_buffer.get_state()
            latent_state = self.model.encode(observation_state).detach()

            episode_reward_sum = 0.0
            episode_observations = [observation]
            for episode_frame_number in range(max_episode_length):

                action = self.policy.select_eval_action(latent_state)

                next_observation, reward, done, info = self.env.step(action)

                episode_reward_sum += reward

                self.frame_buffer.append(next_observation)
                observation_state = self.frame_buffer.get_state()
                latent_state = self.model.encode(observation_state).detach()

                episode_observations.append(next_observation)

                if done:
                    break

            avg_reward = episode_reward_sum / (episode_frame_number + 1)
            avg_reward_list.append(avg_reward)
            self.sacred_logger.log_metric(
                name="avg_reward", value=avg_reward, step=episode_number
            )

            info = {
                "episode_number": episode_number,
                "avg_reward": avg_reward,
                "eval_observations": episode_observations,
                "modalities": self.modalities,
            }
            if episode_number % 10 == 0:
                post_episode_cb.on_eval_episode_end(info)

            episode_number += 1

            print(
                f"**** Eval Episode: {episode_number}/{max_episodes}\t"
                f"Episode avg reward: {avg_reward:.3f}"
            )

        # Print Final average score
        print(
            f"**** Final score: "
            f"Agent reward: {np.mean(avg_reward_list):.3f} +- {np.std(avg_reward_list):.3f} "
        )


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0

        transitions = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).unsqueeze(1)
        terminal_batch = (
            torch.tensor(batch.terminal, dtype=torch.float).to(self.device).unsqueeze(1)
        )

        # Update critic
        next_target_actions = self.controller.actor_target(next_state_batch)
        target_next_state_action_values = self.controller.critic_target(
            [next_state_batch, next_target_actions]
        ).detach()
        target_state_action_values = (
            reward_batch
            + self.gamma * (1.0 - terminal_batch) * target_next_state_action_values
        )
        state_action_values = self.controller.critic([state_batch, action_batch])

        critic_loss = F.mse_loss(
            state_action_values.double(),
            target_state_action_values.double(),
            reduction="mean",
        )
        self.controller.critic.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        self.critic_optim.step()

        # Update actor
        actor_loss = -self.controller.critic(
            [state_batch, self.controller.actor(state_batch)]
        ).mean()
        self.controller.actor.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        self.actor_optim.step()

        soft_update(self.controller.actor_target, self.controller.actor, self.tau)
        soft_update(self.controller.critic_target, self.controller.critic, self.tau)

        return critic_loss.cpu().data.numpy(), actor_loss.cpu().data.numpy()