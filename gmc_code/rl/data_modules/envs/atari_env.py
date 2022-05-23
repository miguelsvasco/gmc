import gym
import gmc_code.rl.data_modules.envs.pendulum_env as ps

class AtariEnv(gym.Env):
    def __init__(self,
                 scenario,
                 scenario_cfg = None,
                 seed=0):

        # Variables
        self._scenario = scenario
        self._scenario_cfg = scenario_cfg
        self._seed = seed

        # Load environment
        if scenario == 'pendulum':
            if scenario_cfg is None:
                self._scenario_cfg = {'sound_frequency': 440.,
                                      'sound_velocity': 20.,
                                      'sound_receivers': ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']}

            self._env = ps.PendulumSound(original_frequency=scenario_cfg['sound_frequency'],
                                         sound_vel=scenario_cfg['sound_velocity'],
                                         sound_receivers=[ps.SoundReceiver(ps.SoundReceiver.Location[ss]) for ss in
                                                          scenario_cfg['sound_receivers']])

        else:
            raise ValueError('Incorrect initialization of Trust Environment: ' + str(scenario))

        # Gym variables
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self._env.seed(self._seed)

        # Recording components
        self._save_data = False
        self._env_data = {'image':[], 'sound':[]}


    def step(self, action):

        observation, reward, done, info = self._env.step(action)

        if self._save_data:
            self._env_data['image'].append(observation[0])
            self._env_data['sound'].append(observation[1])

        return observation, reward, done, info

    def reset(self, **kwargs):
        return self._env.reset()

    def render(self, mode='human', close=False):
        self._env.render()

    def get_n_actions(self):
        return self._env.action_space.shape[0]

    def get_scenario(self):
        return self._scenario

    def get_save_data(self):
        return self._env_data

    def set_save_data(self, bool):
        self._save_data = bool

    def clear_save_data(self):
        self._env_data = {'image': [],
                          'sound': []}



