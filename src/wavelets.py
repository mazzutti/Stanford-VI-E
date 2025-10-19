# src/wavelets.py
import numpy as np


def ricker_wavelet(f_peak, length=0.128, dt=0.002):
    """
    Generates a Ricker wavelet.

    Args:
        f_peak (float): The peak frequency of the wavelet in Hz.
        length (float): The total length of the wavelet in seconds.
        dt (float): The time sampling interval in seconds.

    Returns:
        np.ndarray: A 1D array representing the wavelet.
    """
    t = np.arange(-length / 2, length / 2, dt)
    pi_sq = np.pi**2
    f_sq = f_peak**2
    t_sq = t**2

    # Ricker wavelet formula
    term1 = 1 - 2 * pi_sq * f_sq * t_sq
    term2 = np.exp(-pi_sq * f_sq * t_sq)

    return term1 * term2
