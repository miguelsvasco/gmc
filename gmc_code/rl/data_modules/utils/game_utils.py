import numpy as np


# Environments utils
def pendulum_image_preprocess(observation):
    processed_observation = observation.copy()
    # crop
    processed_observation = processed_observation[20:80, 20:80]

    # downsample
    # processed_observation = processed_observation[::3, ::3, :]
    # remove color
    processed_observation = processed_observation[:, :, 0]

    # make rod white, backround black
    should_be_white_indexes = (processed_observation == 0) | (
        processed_observation == 204)
    should_be_black_indexes = (processed_observation == 255)
    processed_observation[should_be_white_indexes] = 1
    processed_observation[should_be_black_indexes] = 0

    return processed_observation

def pendulum_sound_preprocess(observation, sound_norm):

    # Sound Preprocess
    processed_observation = np.array(observation)

    # preprocess image the same way as the dataset
    min_freq, max_freq = sound_norm['frequency']
    processed_observation[:, 0] = (processed_observation[:, 0] - min_freq) / (max_freq - min_freq)

    min_amp, max_amp = sound_norm['amplitude']
    processed_observation[:, 1] = (processed_observation[:, 1] - min_amp) / (max_amp - min_amp)

    return processed_observation



def modified_doppler_effect(freq, obs_pos, obs_vel, obs_speed, src_pos,
                            src_vel, src_speed, sound_vel):
    # Normalize velocity vectors to find their directions (zero values
    # have no direction).
    if not np.all(src_vel == 0):
        src_vel = src_vel / np.linalg.norm(src_vel)
    if not np.all(obs_vel == 0):
        obs_vel = obs_vel / np.linalg.norm(obs_vel)

    src_to_obs = obs_pos - src_pos
    obs_to_src = src_pos - obs_pos
    if not np.all(src_to_obs == 0):
        src_to_obs = src_to_obs / np.linalg.norm(src_to_obs)
    if not np.all(obs_to_src == 0):
        obs_to_src = obs_to_src / np.linalg.norm(obs_to_src)

    src_radial_vel = src_speed * src_vel.dot(src_to_obs)
    obs_radial_vel = obs_speed * obs_vel.dot(obs_to_src)

    fp = ((sound_vel + obs_radial_vel) / (sound_vel - src_radial_vel)) * freq

    return fp


def inverse_square_law_observer_receiver(obs_pos, src_pos, K=1.0, eps=0.0):
    """
    Computes the inverse square law for an observer receiver pair.
    Follows https://en.wikipedia.org/wiki/Inverse-square_law
    """
    distance = np.linalg.norm(obs_pos - src_pos)
    return K * 1.0 / (distance**2 + eps)