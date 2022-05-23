import imageio
import argparse
import numpy as np
from logging import getLogger

try:
    import sacred
except ImportError:
    raise ImportError("Missing sacred package.  Run `pip install sacred`")

from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_only

logger = getLogger(__name__)


class SacredLogger(LightningLoggerBase):
    def __init__(self, sacred_experiment):
        """Initialize a sacred logger.
        :param sacred.experiment.Experiment sacred_experiment: Required. Experiment object with desired observers
        already appended.
        """
        super().__init__()
        self.sacred_experiment = sacred_experiment
        self.experiment_name = sacred_experiment.path
        self._run_id = None

    @property
    def experiment(self):
        return self.sacred_experiment

    @property
    def run_id(self):
        if self._run_id is not None:
            return self._run_id

        self._run_id = self.sacred_experiment.current_run._id
        return self._run_id

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        # probably not needed bc. it is dealt with by sacred
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning(f"Discarding metric with string value {k}={v}")
                continue
            self.experiment.log_scalar(k, v, step)

    @rank_zero_only
    def log_metric(self, name, value, step=None):
        if isinstance(value, str):
            logger.warning(f"Discarding metric with string value {name}={value}")
            return
        self.experiment.log_scalar(name, value, step)

    @rank_zero_only
    def log_artifact(self, name, filepath):
        self.experiment.add_artifact(filepath, name=name)

    def log_rl_gif(self, observations, filepath, name):

        frame_rate = 27  # fps

        # 1st ep, 1st frame, image
        frame_shape = observations[0][0][0].shape
        black_frames = [
            np.zeros(frame_shape, dtype=np.uint8) for _ in range(frame_rate)
        ]
        white_frames = [
            np.ones(frame_shape, dtype=np.uint8) * 255 for _ in range(2 * frame_rate)
        ]

        # Create Gif
        frames_for_gif = []
        for episode_observations in observations:
            frames_for_gif.extend([image for (image, sound) in episode_observations])
            frames_for_gif.extend(black_frames)
        frames_for_gif.extend(white_frames)

        # Save Gif
        imageio.mimsave(
            filepath, frames_for_gif, duration=1.0 / frame_rate, subrectangles=True
        )

        self.experiment.add_artifact(filepath, name=name)

    def log_rl_eval_gif(self, observations, filepath, name):

        frame_rate = 27  # fps

        # Create Gif
        frames_for_gif = []
        for episode_observations in observations:
            image = episode_observations[0]
            frames_for_gif.extend([image])

        # Save Gif
        imageio.mimsave(
            filepath, frames_for_gif, duration=1.0 / frame_rate, subrectangles=True
        )

        self.experiment.add_artifact(filepath, name=name)

    @property
    def name(self):
        return self.experiment_name

    @property
    def version(self):
        return self.run_id
