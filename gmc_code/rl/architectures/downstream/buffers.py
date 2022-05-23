import random
import torch
import numpy as np
from collections import deque
from gmc_code.rl.utils.rl_utils import Transition


class FrameBuffer:
    """A circular buffer implemented as a deque to keep track of the last few
    frames in the environment that together form a state capturing temporal
    and directional information. Provides an accessor to get the current
    state at any given time, which is represented as a list of consecutive
    frames.

    Also takes pre/post-processors to potentially resize or modify the frames
    before inserting them into the buffer."""

    def __init__(self,
                 frames_per_state,
                 processor,
                 eval_mode=False):
        """
        @param mods:              Number of modalities of the environment.
        @param frames_per_state:  Number of consecutive frames that form a state.
        @param preprocessor:      Lambda that takes a frame and returns a
                                  preprocessed frame.
        """
        if frames_per_state <= 0:
            raise RuntimeError('Frames per state should be greater than 0')

        self.processor = processor
        self.mods = processor.get_n_mods()
        self.frames_per_state = frames_per_state
        self.samples = []
        for _ in range(self.mods):
            self.samples.append(deque(maxlen=frames_per_state))

        self.processor = processor
        self.eval_mode = eval_mode

    def append(self, sample):
        """
        Takes a frame, applies preprocessing, and appends it to the deque.
        The first frame added to the buffer is duplicated `frames_per_state` times
        to completely fill the buffer.
        """

        sample = self.processor.preprocess(sample)
        if len(self.samples[0]) == 0:
            for i in range(self.mods):
                self.samples[i].extend(self.frames_per_state * [sample[i]])
        for i in range(self.mods):
            self.samples[i].append(sample[i])

    def get_state(self):
        """
        Fetch the current state consisting of `frames_per_state` consecutive frames.
        If `frames_per_state` is 1, returns the frame instead of an array of
        length 1. Otherwise, returns a Numpy array of `frames_per_state` frames.
        """
        if len(self.samples[0]) == 0:
            return None
        elif self.frames_per_state == 1:
            post_samples = []
            for i in range(self.mods):
                post_samples.append(self.samples[i][0])
        else:
            post_samples = []
            for i in range(self.mods):
                post_samples.append(list(self.samples[i]))
        if not self.eval_mode:
            return self.processor.postprocess(list(post_samples))
        else:
            return self.processor.eval_postprocess(list(post_samples))

    def reset(self):
        for i in range(self.mods):
            self.samples[i].clear()


# Replay Memories
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperhot Replay Memory
class HyperhotReplayMemory(ReplayMemory):
    def __init__(self, capacity):
        super(HyperhotReplayMemory, self).__init__(capacity)
        self._running_reward_sum = 0.0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        transition = Transition(*args)
        self._update_running_reward_sum(transition.reward.cpu().item())
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def _update_running_reward_sum(self, new_reward):
        self._running_reward_sum += new_reward
        if self.memory[self.position] is not None:
            old_reward = self.memory[self.position].reward.cpu().item()
            self._running_reward_sum -= old_reward

    def avg_reward(self):
        return self._running_reward_sum / len(self.memory)


# Pendulum Replay Memory
class PendulumReplayMemory(ReplayMemory):
    def __init__(self, capacity):
        super(PendulumReplayMemory, self).__init__(capacity)

    def stats(self):
        unrolled = Transition(*zip(*self.memory))
        reward = torch.cat(unrolled.reward).cpu().numpy()
        return np.mean(reward), np.std(reward)
