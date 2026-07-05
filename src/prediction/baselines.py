"""Non-learned baselines. Constant velocity is the bar every model must beat."""
import numpy as np


def constant_velocity(obs: np.ndarray, pred_len: int) -> np.ndarray:
    """Extrapolate with the mean velocity of the observed window.

    obs: (T_obs, 2) observed positions in meters.
    Returns (pred_len, 2) future positions.
    Mean velocity over the window is more robust than last-step velocity,
    which is noisy for tracked (non-GT) data.
    """
    v = (obs[-1] - obs[0]) / (len(obs) - 1)
    steps = np.arange(1, pred_len + 1)[:, None]
    return obs[-1] + steps * v