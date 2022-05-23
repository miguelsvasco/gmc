import os
import gym
import errno
import torch
import numpy as np
from collections import deque
import gmc_code.rl.data_modules.envs.pendulum_env as ps
from gmc_code.rl.data_modules.utils.game_utils import pendulum_image_preprocess

def _random_action(env):
    if type(env.action_space) is gym.spaces.discrete.Discrete:  # For discrete Action Spaces
        return env.action_space.sample()
    else:                                                       # For continuous Action Spaces
        return np.random.uniform(env.action_space.low, env.action_space.high)

def generate_dataset_filename(scenario, scenario_cfg):
    if scenario == 'pendulum':

        snd_f = scenario_cfg['sound_frequency']
        snd_vel = scenario_cfg['sound_velocity']
        snd_rcv = scenario_cfg['sound_receivers']

        train_filename = '_'.join([
            f'train_pendulum_dataset_samples{scenario_cfg["train_samples"]}', f'stack{scenario_cfg["n_stack"]}',
            f'freq{snd_f}', f'vel{snd_vel}',
            f'rec{str(snd_rcv)}.pt'])

        test_filename = '_'.join([
            f'test_pendulum_dataset_samples{scenario_cfg["test_samples"]}', f'stack{scenario_cfg["n_stack"]}',
            f'freq{snd_f}', f'vel{snd_vel}',
            f'rec{str(snd_rcv)}.pt'])

    else:
        raise ValueError('Incorrect initialization of Atari Game Scenario.')

    return train_filename, test_filename


def generate_dataset(scenario, data_dir, scenario_cfg):

    if scenario == 'pendulum':
        env = ps.PendulumSound(
            original_frequency=scenario_cfg['sound_frequency'],
            sound_vel=scenario_cfg['sound_velocity'],
            sound_receivers=[
                ps.SoundReceiver(ps.SoundReceiver.Location[ss])
                for ss in scenario_cfg['sound_receivers']
            ])
    else:
        raise ValueError("Incorrect initialization of Atari Games scenario: " + str(scenario))

    # Setup environment
    env.seed(scenario_cfg['random_seed'])
    np.random.seed(scenario_cfg['random_seed'])
    train_filename, test_filename = generate_dataset_filename(scenario, scenario_cfg)

    # Setup Frame Buffer to hold observations
    frame_buffer = DatasetFrameBuffer(env_name=scenario, frames_per_state=scenario_cfg['n_stack'])
    train_samples = scenario_cfg['train_samples']
    test_samples = scenario_cfg['test_samples']


    # Train Dataset

    frame_number = 0
    episode_number = 0
    img_state_lst = []
    snd_state_lst = []
    action_lst = []
    done_lst = []
    reward_lst = []
    next_img_state_lst = []
    next_snd_state_lst = []
    new_episode = True

    while frame_number < train_samples:
        if new_episode:
            frame_buffer.reset()
            observation = env.reset()
            frame_buffer.append(observation)
            img_state, snd_state = frame_buffer.get_state()

            print(
                f'Episode: {episode_number} - {frame_number}/{train_samples}'
            )
            new_episode = False

        action = _random_action(env)
        next_observation, reward, done, info = env.step(action)

        frame_buffer.append(next_observation)
        next_img_state, next_snd_state = frame_buffer.get_state()

        torch_action = torch.from_numpy(action)
        torch_reward = torch.tensor([reward])

        # Append information
        img_state_lst.append(img_state)
        snd_state_lst.append(snd_state)
        action_lst.append(torch_action)
        reward_lst.append(torch_reward)
        next_img_state_lst.append(next_img_state)
        next_snd_state_lst.append(next_snd_state)

        # Update
        img_state = next_img_state
        snd_state = next_snd_state

        if done:
            done_lst.append(1.0)
            episode_number += 1
            new_episode = True
        else:
            done_lst.append(0.0)

        frame_number += 1

    # Image State and Image Next State
    image_state = np.stack(img_state_lst)
    next_image_state = np.stack(next_img_state_lst)
    t_images = torch.from_numpy(image_state).float()
    t_next_images = torch.from_numpy(next_image_state).float()

    # Sound State and Sound Next State
    sound_states = np.stack(snd_state_lst)
    next_sound_states = np.stack(next_snd_state_lst)

    if scenario == 'pendulum':

        # normalize frequencies
        max_freq, min_freq = np.max(sound_states[:, :, :, 0]), np.min(sound_states[:, :, :, 0])
        sound_states[:, :, :, 0] = (sound_states[:, :, :, 0] - min_freq) / (
                max_freq - min_freq)

        # normalize amplitudes
        max_amp, min_amp = np.max(sound_states[:, :, :, 1]), np.min(sound_states[:, :, :, 1])
        sound_states[:, :, :, 1] = (sound_states[:, :, :, 1] - min_amp) / (max_amp - min_amp)

        sound_normalization_info = {
            'frequency': (min_freq, max_freq),
            'amplitude': (min_amp, max_amp)
        }

        next_sound_states[:, :, :, 0] = (next_sound_states[:, :, :, 0] - min_freq) / (max_freq - min_freq)
        next_sound_states[:, :, :, 1] = (next_sound_states[:, :, :, 1] - min_amp) / (max_amp - min_amp)

    else:
        raise ValueError("Incorrect initialization of Atari Games scenario: " + str(scenario))


    t_sounds = torch.from_numpy(sound_states).float()
    t_next_sounds = torch.from_numpy(next_sound_states).float()

    # Actions, Reward and Done
    t_actions = torch.from_numpy(np.stack(action_lst)).float()
    t_reward = torch.from_numpy(np.stack(reward_lst)).float()
    t_done = torch.from_numpy(np.stack(done_lst)).float()


    try:
        os.makedirs(data_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    with open(os.path.join(data_dir, train_filename),'wb') as f:
        torch.save((t_images, t_sounds, t_actions, t_reward, t_next_images, t_next_sounds, t_done,
                    sound_normalization_info), f)


    # Test Dataset

    frame_number = 0
    episode_number = 0
    img_state_lst = []
    snd_state_lst = []
    action_lst = []
    done_lst = []
    reward_lst = []
    next_img_state_lst = []
    next_snd_state_lst = []
    new_episode = True

    while frame_number < test_samples:
        if new_episode:
            frame_buffer.reset()
            observation = env.reset()
            frame_buffer.append(observation)
            img_state, snd_state = frame_buffer.get_state()

            print(
                f'Episode: {episode_number} - {frame_number}/{test_samples}'
            )
            new_episode = False

        action = _random_action(env)
        next_observation, reward, done, info = env.step(action)

        frame_buffer.append(next_observation)
        next_img_state, next_snd_state = frame_buffer.get_state()
        torch_action = torch.from_numpy(action)
        torch_reward = torch.tensor([reward])

        # Append information
        img_state_lst.append(img_state)
        snd_state_lst.append(snd_state)
        action_lst.append(torch_action)
        reward_lst.append(torch_reward)
        next_img_state_lst.append(next_img_state)
        next_snd_state_lst.append(next_snd_state)

        # Update
        img_state = next_img_state
        snd_state = next_snd_state

        if done:
            done_lst.append(1.0)
            episode_number += 1
            new_episode = True
        else:
            done_lst.append(0.0)

        frame_number += 1

    # Image State and Image Next State
    image_state = np.stack(img_state_lst)
    next_image_state = np.stack(next_img_state_lst)
    t_images = torch.from_numpy(image_state).float()
    t_next_images = torch.from_numpy(next_image_state).float()

    # Sound State and Sound Next State
    sound_states = np.stack(snd_state_lst)
    next_sound_states = np.stack(next_snd_state_lst)

    if scenario == 'pendulum':

        # normalize frequencies
        max_freq, min_freq = np.max(sound_states[:, :, :, 0]), np.min(sound_states[:, :, :, 0])
        sound_states[:, :, :, 0] = (sound_states[:, :, :, 0] - min_freq) / (
                max_freq - min_freq)

        # normalize amplitudes
        max_amp, min_amp = np.max(sound_states[:, :, :, 1]), np.min(sound_states[:, :, :, 1])
        sound_states[:, :, :, 1] = (sound_states[:, :, :, 1] - min_amp) / (max_amp - min_amp)

        sound_normalization_info = {
            'frequency': (min_freq, max_freq),
            'amplitude': (min_amp, max_amp)
        }

        next_sound_states[:, :, :, 0] = (next_sound_states[:, :, :, 0] - min_freq) / (max_freq - min_freq)
        next_sound_states[:, :, :, 1] = (next_sound_states[:, :, :, 1] - min_amp) / (max_amp - min_amp)

    else:
        raise ValueError("Incorrect initialization of Atari Games scenario: " + str(env_name))

    t_sounds = torch.from_numpy(sound_states).float()
    t_next_sounds = torch.from_numpy(next_sound_states).float()

    # Actions, Reward and Done
    t_actions = torch.from_numpy(np.stack(action_lst)).float()
    t_reward = torch.from_numpy(np.stack(reward_lst)).float()
    t_done = torch.from_numpy(np.stack(done_lst)).float()

    try:
        os.makedirs(data_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    with open(os.path.join(data_dir, test_filename), 'wb') as f:
        torch.save((t_images, t_sounds, t_actions, t_reward, t_next_images, t_next_sounds, t_done,
                    sound_normalization_info), f)


    env.close()
    return



class DatasetFrameBuffer:
    """A circular buffer implemented as a deque to keep track of the last few
    frames in the environment that together form a state capturing temporal
    and directional information. Provides an accessor to get the current
    state at any given time, which is represented as a list of consecutive
    frames.

    Also takes pre/post-processors to potentially resize or modify the frames
    before inserting them into the buffer."""

    def __init__(self, env_name, frames_per_state, postprocessor=lambda x: np.stack(x, axis=0)):
        """
        @param frames_per_state:         Number of consecutive frames that form a state.
        @param sound normalization:      Sound Normalization.
        """
        self.env_name = env_name
        if frames_per_state <= 0:
            raise RuntimeError('Frames per state should be greater than 0')

        self.frames_per_state = frames_per_state
        self.img_samples = deque(maxlen=frames_per_state)
        self.snd_samples = deque(maxlen=frames_per_state)
        self.postprocessor = postprocessor

    def append(self, sample):
        """
        Takes a frame, applies preprocessing, and appends it to the deque.
        The first frame added to the buffer is duplicated `frames_per_state` times
        to completely fill the buffer.
        """
        img_sample, snd_sample = sample

        # Preprocess image
        if self.env_name == 'pendulum':
            img_sample = pendulum_image_preprocess(img_sample)
        else:
            raise ValueError('Incorrect initialization of the Multimodal Atari Game scenario: ' + str(self.env_name))

        if len(self.img_samples) == 0:
            self.img_samples.extend(self.frames_per_state * [img_sample])
            self.snd_samples.extend(self.frames_per_state * [snd_sample])
        self.img_samples.append(img_sample)
        self.snd_samples.append(snd_sample)

    def get_state(self):
        """
        Fetch the current state consisting of `frames_per_state` consecutive frames.
        If `frames_per_state` is 1, returns the frame instead of an array of
        length 1. Otherwise, returns a Numpy array of `frames_per_state` frames.
        """
        if len(self.img_samples) == 0:
            return None
        if self.frames_per_state == 1:
            post_img = self.postprocessor([self.img_samples[0]])
            post_snd = self.postprocessor([self.snd_samples[0]])
            return post_img, post_snd

        post_img = self.postprocessor(list(self.img_samples))
        post_snd = self.postprocessor(list(self.snd_samples))

        return post_img, post_snd

    def reset(self):
        self.img_samples.clear()
        self.snd_samples.clear()