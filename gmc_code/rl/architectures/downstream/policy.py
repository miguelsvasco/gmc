import torch
import numpy as np
from gmc_code.rl.utils.rl_utils import OUNoise


class Policy(object):
    def __init__(self, policy_net, action_space, controller_config):
        self.policy_net = policy_net
        self.action_space = action_space
        self.controller_config = controller_config

    def select_action(self, state, frame_number, eval=False):
        return

    def random_action(self):
        self.action_space.sample()


class HyperhotPolicy(Policy):       # Epsilon-Greedy
    def __init__(self, policy_net, action_space, controller_config):
        super(HyperhotPolicy, self).__init__(policy_net, action_space, controller_config)

        self.eps_initial = controller_config['policy_config']['eps_initial']
        self.eps_end = controller_config['policy_config']['eps_end']
        self.n_annealing_frames = controller_config['policy_config']['n_annealing_frames']
        self.replay_buffer_start_size = controller_config['policy_config']['replay_buffer_start_size']
        self.eps_evaluation = controller_config['policy_config']['eps_evaluation']

    def select_action(self, state, frame_number, evaluation=False):
        if evaluation:
            eps = self.eps_evaluation
        elif frame_number < self.replay_buffer_start_size:
            eps = 1.0
        else:
            eps = self.eps_initial + (self.eps_end - self.eps_initial) * (
                    frame_number -
                    self.replay_buffer_start_size) / self.n_annealing_frames
            eps = max(eps, self.eps_end)

        if np.random.rand(1) > eps:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected
                # reward.
                return self.policy_net(state).max(1)[1].detach().cpu().numpy()
        else:
            return self.random_action()

    def random_action(self):
        return np.array([self.action_space.sample()])


class PendulumPolicy(Policy):
    def __init__(self, policy_net, action_space, controller_config):
        super(PendulumPolicy, self).__init__(policy_net, action_space, controller_config)

        self.replay_buffer_size = controller_config['memory_size']
        self.random_process = OUNoise(action_space=action_space,
                                      mu=controller_config['policy_config']['ou_mu'],
                                      theta=controller_config['policy_config']['ou_theta'],
                                      max_sigma=controller_config['policy_config']['ou_max_sigma'],
                                      min_sigma=controller_config['policy_config']['ou_min_sigma'],
                                      decay_period=controller_config['policy_config']['ou_decay_period'])

    def random_action(self):
        return self.action_space.sample()

    def select_action(self, state, frame_number, evaluation=False):

        if evaluation or (frame_number >= self.replay_buffer_size):
            with torch.no_grad():
                action = self.policy_net(state).squeeze(0).detach().cpu().numpy()
            action += (not evaluation) * self.random_process.get_action(action, frame_number).squeeze(0)
            action = np.clip(action, self.action_space.low, self.action_space.high)
        else:
            action = self.random_action()

        return action

    def select_eval_action(self, state):
        with torch.no_grad():
            action = self.policy_net(state).squeeze(0).detach().cpu().numpy()
            return np.clip(action, self.action_space.low, self.action_space.high)

