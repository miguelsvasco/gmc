import pickle
import pyglet
import numpy as np
from enum import Enum
from pysine import sine
from gym.envs.classic_control import rendering
from gym.envs.classic_control.pendulum import PendulumEnv
from gmc_code.rl.data_modules.utils.game_utils import modified_doppler_effect, inverse_square_law_observer_receiver

# TODO(rsilva): This should go inside PendulumSound, but it seems we
# are getting circular dependencies when we do so
BOTTOM_MARGIN = -2.2
TOP_MARGIN = 2.2
LEFT_MARGIN = 2.2
RIGHT_MARGIN = -2.2


class CustomViewer(rendering.Viewer):
    def __init__(self, width, height, display=None):
        super().__init__(width, height, display=None)
        self.window = pyglet.window.Window(width=width, height=height, display=display, vsync=False)


class SoundReceiver(object):
    class Location(Enum):
        LEFT_BOTTOM = 1,
        LEFT_MIDDLE = 2,
        LEFT_TOP = 3,
        RIGHT_TOP = 4,
        RIGHT_MIDDLE = 5,
        RIGHT_BOTTOM = 6,
        MIDDLE_TOP = 7,
        MIDDLE_BOTTOM = 8

    def __init__(self, location):
        self.location = location

        if location == SoundReceiver.Location.LEFT_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, LEFT_MARGIN])
        elif location == SoundReceiver.Location.LEFT_MIDDLE:
            self.pos = np.array([0.0, LEFT_MARGIN])
        elif location == SoundReceiver.Location.LEFT_TOP:
            self.pos = np.array([TOP_MARGIN, LEFT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_TOP:
            self.pos = np.array([TOP_MARGIN, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_MIDDLE:
            self.pos = np.array([0.0, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.RIGHT_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, RIGHT_MARGIN])
        elif location == SoundReceiver.Location.MIDDLE_TOP:
            self.pos = np.array([TOP_MARGIN, 0.0])
        elif location == SoundReceiver.Location.MIDDLE_BOTTOM:
            self.pos = np.array([BOTTOM_MARGIN, 0.0])


class PendulumSound(PendulumEnv):
    """
    Frame:
    - points stored as (height, weight)
    - positive upwards and left
    Angular velocity:
    - positive is ccw
    """

    def __init__(
            self,
            original_frequency=440.,
            sound_vel=20.,
            sound_receivers=[SoundReceiver(SoundReceiver.Location.RIGHT_TOP)],
            debug=False):
        super().__init__()
        self.original_frequency = original_frequency
        self.sound_vel = sound_vel
        self.sound_receivers = sound_receivers
        self._debug = debug

        self.reset()

    def step(self, a):
        observation, reward, done, info = super().step(a)

        x, y, thdot = observation
        abs_src_vel = np.abs(thdot * 1)  # v = w . r
        # compute ccw perpendicular vector. if angular velocity is
        # negative, we reverse it. then multiply by absolute velocity
        src_vel = np.array([-y, x])
        src_vel = (
            src_vel / np.linalg.norm(src_vel)) * np.sign(thdot) * abs_src_vel
        src_pos = np.array([x, y])

        self._frequencies = [
            modified_doppler_effect(
                self.original_frequency,
                obs_pos=rec.pos,
                obs_vel=np.zeros(2),
                obs_speed=0.0,
                src_pos=src_pos,
                src_vel=src_vel,
                src_speed=np.linalg.norm(src_vel),
                sound_vel=self.sound_vel) for rec in self.sound_receivers
        ]
        self._amplitudes = [
            inverse_square_law_observer_receiver(
                obs_pos=rec.pos, src_pos=src_pos)
            for rec in self.sound_receivers
        ]
        sound_observation = list(zip(self._frequencies, self._amplitudes))

        img_observation = self.render(mode='rgb_array')

        if self._debug:
            self._debug_data['pos'].append(src_pos)
            self._debug_data['vel'].append(src_vel)
            self._debug_data['sound'].append(self._frequencies)

        return (img_observation, sound_observation), reward, done, info

    def render(self, mode='human', sound_channel=0, sound_duration=.1):
        if self.viewer is None:
            self.viewer = CustomViewer(100, 100)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)

        # only play sound in human mode
        if self._frequencies[sound_channel] and (mode == 'human'):
            sine(
                frequency=self._frequencies[sound_channel],
                duration=sound_duration)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def reset(self, num_initial_steps=1):
        observation = super().reset()

        if self._debug:
            self._debug_data = {
                'pos': [],
                'vel': [],
                'sound_receivers': [rec.pos for rec in self.sound_receivers],
                'sound': []
            }

        if type(num_initial_steps) is list or type(num_initial_steps) is tuple:
            assert len(num_initial_steps) == 2
            low = num_initial_steps[0]
            high = num_initial_steps[1]
            num_initial_steps = np.random.randint(low, high)
        elif type(num_initial_steps) is int:
            assert num_initial_steps >= 1
        else:
            raise 'Unsupported type for num_initial_steps. Either list/tuple or int'

        for _ in range(num_initial_steps):
            (observation, sound), _, _, _ = self.step(np.array([0.0]))

        return observation, sound

    def close(self, out=None):
        super().close()

        if out:
            with open(out, 'wb') as filehandle:
                pickle.dump(
                    self._debug_data,
                    filehandle,
                    protocol=pickle.HIGHEST_PROTOCOL)


def main():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import argparse

    parser = argparse.ArgumentParser(description='PongSound debugger')
    parser.add_argument(
        '--file', type=str, required=True, help='File with debug data')
    args = parser.parse_args()

    COLORS = ['forestgreen', 'cornflowerblue', 'darkorange', 'm']

    # Load data
    debug = pickle.load(open(args.file, 'rb'))
    positions = np.vstack(debug['pos'])
    velocities = np.vstack(debug['vel'])
    sounds = np.vstack(debug['sound'])

    sound_receiver_positions = debug['sound_receivers']
    sound_receiver_positions = np.vstack(sound_receiver_positions)
    n_sound_receivers = sound_receiver_positions.shape[0]

    # Plots
    fig, _ = plt.subplots()

    # - Plot ball data
    ax = plt.subplot('311')
    plt.xlim(LEFT_MARGIN, RIGHT_MARGIN)
    plt.ylim(BOTTOM_MARGIN, TOP_MARGIN)

    # -- Plot ball position
    plt.scatter(positions[:, 1], positions[:, 0], s=3, c='k')
    ball_plot, = plt.plot(positions[0, 1], positions[0, 0], marker='o')

    # -- Plot ball velocity
    vel_arrow = plt.arrow(
        positions[0, 1],
        positions[0, 0],
        velocities[0, 1],
        velocities[0, 0],
        width=4e-2)

    # -- Plot ball to mic line
    src_mic_plots = []
    for sr in range(n_sound_receivers):
        p, = plt.plot([positions[0, 1], sound_receiver_positions[sr, 1]],
                      [positions[0, 0], sound_receiver_positions[sr, 0]],
                      c=COLORS[sr])
        src_mic_plots.append(p)

    time_slider = Slider(
        plt.axes([0.2, 0.05, 0.65, 0.03]),
        'timestep',
        0,
        len(debug['pos']) - 1,
        valinit=0,
        valstep=1)

    # - Plot sound data
    plt.subplot('312')
    sound_marker_plots = []
    for sr in range(n_sound_receivers):
        plt.plot(sounds[:, sr], c=COLORS[sr])
        p, = plt.plot(1, sounds[1, sr], marker='o')
        sound_marker_plots.append(p)

    plt.subplot('313')
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    plt.plot(speeds)
    speed_marker_plot, = plt.plot(0, speeds[0], marker='o')

    def update(_):
        nonlocal vel_arrow
        timestep = int(time_slider.val)

        ball_position = debug['pos'][timestep]
        ball_plot.set_data(ball_position[1], ball_position[0])

        for sr in range(n_sound_receivers):
            src_mic_plots[sr].set_data(
                [positions[timestep, 1], sound_receiver_positions[sr, 1]],
                [positions[timestep, 0], sound_receiver_positions[sr, 0]])

        vel_arrow.remove()
        vel_arrow = ax.arrow(
            positions[timestep, 1],
            positions[timestep, 0],
            velocities[timestep, 1],
            velocities[timestep, 0],
            width=4e-2)
        for sr in range(n_sound_receivers):
            sound_marker_plots[sr].set_data(timestep, sounds[timestep, sr])

        speed_marker_plot.set_data(timestep, speeds[timestep])

        fig.canvas.draw_idle()

    def arrow_key_image_control(event):
        if event.key == 'left':
            time_slider.set_val(max(time_slider.val - 1, time_slider.valmin))
        elif event.key == 'right':
            time_slider.set_val(min(time_slider.val + 1, time_slider.valmax))

        update(0)

    fig.canvas.mpl_connect('key_release_event', arrow_key_image_control)
    time_slider.on_changed(update)

    plt.show()

if __name__ == "__main__":
    main()
