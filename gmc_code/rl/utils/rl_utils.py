import numpy as np
from collections import namedtuple


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminal")
)


class FixedHorizonAverageMeter(object):
    def __init__(self, horizon):
        self.horizon = horizon
        self.reset()

    def reset(self):
        self.vals = []
        self.position = 0
        self.avg = 0

    def update(self, val):
        if len(self.vals) < self.horizon:
            self.vals.append(None)

        self.vals[self.position] = val
        self.position = (self.position + 1) % self.horizon
        self.avg = np.mean(self.vals)


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(
        self,
        theta,
        mu=0.0,
        sigma=1.0,
        dt=1e-2,
        x0=None,
        size=1,
        sigma_min=None,
        n_steps_annealing=1000,
    ):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing
        )
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        )
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def discount_rewards(rewards, gamma):
    discounted_reward = rewards[-1]
    for t in reversed(range(len(rewards) - 1)):
        discounted_reward = rewards[t] + gamma * discounted_reward

    return discounted_reward


class OUNoise(object):
    def __init__(
        self,
        action_space,
        mu=0.0,
        theta=0.15,
        max_sigma=0.3,
        min_sigma=0.3,
        decay_period=100000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action + ou_state, self.low, self.high)


#
#
# def build_controller(n_states, n_actions, controller_config, cuda):
#     if controller_config['model'] == 'ddpg':
#
#         controller = DDPG(n_states=n_states,
#                     n_actions=n_actions,
#                     hidden1=controller_config['actor_layers_sizes'][0],
#                     hidden2=controller_config['actor_layers_sizes'][1],
#                     cuda=cuda)
#     else:
#         raise ValueError("Wrong Controller selected = " + str(controller_config['model']))
#
#     return controller
#
#
# def save_controller_checkpoint(state,
#                        name,
#                        is_best,
#                        folder='./',
#                        filename=None):
#
#     if filename is None:
#         filename = str(name) + '_controller_checkpoint.pth.tar'
#
#     torch.save(state, os.path.join(folder, filename))
#     if is_best:
#         shutil.copyfile(
#             os.path.join(folder, filename),
#             os.path.join(folder, 'best_' + str(name) + '_controller_model.pth.tar'))
#
# def load_controller_checkpoint(checkpoint_file, use_cuda=False):
#     if use_cuda:
#         checkpoint = torch.load(checkpoint_file)
#     else:
#         checkpoint = torch.load(
#             checkpoint_file, map_location=lambda storage, location: storage)
#
#     controller_config = checkpoint['controller']
#     model = build_controller(n_states=checkpoint['n_states'],
#                              n_actions=checkpoint['n_actions'],
#                              controller_config=controller_config,
#                              cuda=use_cuda)
#
#     model.load_state_dict(checkpoint['state_dict'])
#
#     if use_cuda:
#         model.cuda()
#
#     return model, controller_config
#
#
# def make_controller_training_config(model, controller_config, env_config):
#     if model == 'ddpg':
#         return {'observation_mods': controller_config['observation_mods'],
#                 'max_frames': controller_config['max_frames'],
#                 'gamma': controller_config['gamma'],
#                 'memory_size': controller_config['memory_size'],
#                 'max_episode_length': controller_config['max_episode_length'],
#                 'tau': controller_config['tau'],
#                 'policy_config': controller_config['policy_config'],
#                 'batch_size': controller_config['batch_size'],
#                 'actor_learning_rate': controller_config['actor_learning_rate'],
#                 'critic_learning_rate': controller_config['critic_learning_rate'],
#                 'frames_per_state': env_config['n_stack'],
#                 'eval_frequency': controller_config['eval_frequency'],
#                 'eval_length': controller_config['eval_length']}
#     else:
#         raise ValueError("Incorrect perception training setup selected")
#
#
#
# def retrieve_controller_model(config, rep_model_name, ctr_dep_config, cuda):
#
#     if ctr_dep_config['controller_from_file']:
#         controller, controller_checkpoint = load_controller_checkpoint(ctr_dep_config['controller_file_from_file'], cuda)
#
#     elif ctr_dep_config['controller_from_mongodb']:
#         query = {}
#         query['representation_train'] = config['representation_train']
#         query['pendulum_rl_ingredient'] = config['pendulum_rl_ingredient']
#         query['environment'] = config['environment']
#         env_name = config['environment']['name']
#         query = flatten_dict(query, separator='.', prefix='config')
#         query['experiment.name'] = env_name + '_' + rep_model_name + '_train_controller'
#
#         model_file = mongodb.get_experiment_artifact(ctr_dep_config['controller_file_from_mongodb'], query)
#         if model_file is None:
#             raise ValueError(f'Could not find requested Controller model on mongo database.')
#
#         controller, controller_checkpoint = load_controller_checkpoint(model_file, cuda)
#
#     else:
#         raise ValueError("Error loading RL model.")
#
#     return controller, controller_checkpoint
