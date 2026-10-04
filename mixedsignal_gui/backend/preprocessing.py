"""Shared, versioned model-input power normalization."""

import numpy as np

POWER_NORMALIZATION = "mean_antenna_power_v1"


def normalize_model_input(X, *, complex_iq=False):
    """Normalize (batch, channels, samples) to mean unit antenna power.

    One scale per example preserves relative antenna amplitudes and I/Q phase.
    Padding participates in the power calculation; all-zero examples stay zero.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or not X.shape[1] or not X.shape[2]:
        raise ValueError("Model input must have shape (batch, channels, samples).")
    if complex_iq and X.shape[1] % 2:
        raise ValueError("Complex I/Q input requires paired real/imaginary channels.")
    antennas = X.shape[1] // 2 if complex_iq else X.shape[1]
    power = np.mean(np.sum(X.astype(np.float64) ** 2, axis=1), axis=1) / antennas
    if not np.all(np.isfinite(power)):
        raise ValueError("Model input contains non-finite samples.")
    scale = np.sqrt(np.where(power > 0, power, 1))[:, None, None]
    return (X / scale).astype(np.float32)


def normalize_for_model(X, metadata):
    """Honor new checkpoint preprocessing without changing legacy models."""
    config = (metadata or {}).get("power_normalization")
    if not config:
        return X
    if config.get("method") != POWER_NORMALIZATION:
        raise ValueError("Unsupported model power-normalization method.")
    return normalize_model_input(X, complex_iq=config["complex_iq"])
