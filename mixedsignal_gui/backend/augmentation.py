"""
RF Signal Augmentation Pipeline for Machine Learning
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from fractions import Fraction
from scipy.signal import hilbert, resample_poly
import copy
import os

import numpy as np

try:
    import scipy.io as sio
except ImportError:
    sio = None


class AugmentationBlock(ABC):
    """Base class for all augmentation blocks"""

    @abstractmethod
    def apply(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        """
        Apply augmentation to a signal.

        Args:
            signal: Input signal (real passband from MATLAB generator)
            fs: Sample rate in Hz
            **kwargs: Additional parameters

        Returns:
            Augmented signal
        """
        pass

    def __call__(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        """Allow calling block as a function"""
        return self.apply(signal, fs, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AugmentationPipeline:
    """class to chain multiple augmentation blocks together"""

    def __init__(self, blocks: Optional[List[AugmentationBlock]] = None):
        self.blocks = blocks if blocks is not None else []

    def add(self, block: AugmentationBlock) -> 'AugmentationPipeline':
        self.blocks.append(block)
        return self

    def apply(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        """
        Apply all blocks in sequence.

        Returns augmented signal after all blocks
        """
        result = signal.copy()
        for block in self.blocks:
            result = block(result, fs, **kwargs)
        return result

    def __call__(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        return self.apply(signal, fs, **kwargs)

    def __repr__(self) -> str:
        block_names = [repr(b) for b in self.blocks]
        return f"AugmentationPipeline([{', '.join(block_names)}])"


# Additive White Gaussian Noise
class AWGNAugmentation(AugmentationBlock):
    def __init__(
        self,
        snr_db: float = 20.0,
        seed: Optional[int] = None
    ):
        """
            snr_db: Fixed SNR in dB (used if snr_range is None)
        """
        self.snr_db = snr_db
        self.rng = np.random.default_rng(seed)

    def apply(self, signal: np.ndarray, _fs: float, **_kwargs) -> np.ndarray:
        """
        Add AWGN to the signal.

        Args:
            signal: Input signal (real or complex)
            _fs: Sample rate (not used for amplitude augmentation)

        Returns:
            Signal with added noise at specified SNR
        """
        # Determine SNR to use
        snr_db = self.snr_db
        
        sig_power = np.mean(np.abs(signal) ** 2)

        snr_linear = 10 ** (snr_db / 10)
        noise_power = sig_power / snr_linear

        # Generate noise based on signal type
        if np.iscomplexobj(signal):
            # Complex: noise power split between I and Q
            noise_std = np.sqrt(noise_power / 2)
            noise = (self.rng.normal(0, noise_std, signal.shape) +
                     1j * self.rng.normal(0, noise_std, signal.shape))
        else:
            noise_std = np.sqrt(noise_power)
            noise = self.rng.normal(0, noise_std, signal.shape)

        return signal + noise

    def __repr__(self) -> str:
        return f"AWGNAugmentation(snr_db={self.snr_db})"

class ScalarAmplitudeAndPhaseShift(AugmentationBlock):
    '''
    This block allows you to modulate the amplitude and/or the phase of a signal by
    multiplying the signal by a complex scalar

    amplitude: scalar to amplify or attenuate magnitude of signal
    phi: radians to delay signal by
    '''


    def __init__(
        self,
        amplitude: float = 1.0,
        phi: float = 0.0,
    ):
        self.amplitude = amplitude
        self.phi = phi

    def apply(self, signal: np.ndarray, _fs: float, **_kwargs) -> np.ndarray:
        # build the complex channel scalar  h = |a| * e^(jφ)
        h = self.amplitude * np.exp(1j * self.phi)

        if np.iscomplexobj(signal):
            # Complex baseband: multiply directly
            return (h * signal).astype(signal.dtype)
        else:
            # Real passband: use analytic signal via Hilbert transform
            x_a = hilbert(signal, axis=-1)
            y_a = h * x_a
            return np.real(y_a)

    def __repr__(self) -> str:
        return f"ScalarAmplitudeAndPhaseShift(amplitude={self.amplitude:.4f}, phi={self.phi:.4f})"


class FrequencyShift(AugmentationBlock):
    """Apply a carrier frequency offset (CFO) to the signal.

    Models oscillator mismatch between TX and RX by multiplying by
    ``e^{j 2π Δf t}``, producing a linearly increasing phase that
    causes the constellation to spin over time.

    Parameters
    ----------
    delta_f : float
        Frequency offset in Hz.
    """

    def __init__(self, delta_f: float = 0.0):
        self.delta_f = delta_f

    def apply(self, signal: np.ndarray, fs: float, **_kwargs) -> np.ndarray:
        if self.delta_f == 0.0:
            return signal

        n_samples = signal.shape[-1]
        t = np.arange(n_samples) / fs
        shift = np.exp(1j * 2 * np.pi * self.delta_f * t)

        if np.iscomplexobj(signal):
            return (signal * shift).astype(signal.dtype)
        else:
            # Shift the analytic representation; cosine mixing would create an
            # unwanted image at the sum and difference frequencies.
            analytic = hilbert(signal.astype(np.float64), axis=-1)
            return np.real(analytic * shift).astype(signal.dtype)

    def __repr__(self) -> str:
        return f"FrequencyShift(delta_f={self.delta_f:.1f} Hz)"


# ---------------------------------------------------------------------------
# Stochastic TDL Channel helpers (ported from CLIP_datagen/Stochastic)
# ---------------------------------------------------------------------------

def _load_waveform(path: str,
                   fmt: str,
                   mat_key: Optional[str],
                   iq_layout: str = "auto") -> np.ndarray:
    """
    Returns a 1-D complex numpy array.
    Supports:
      - .npy: complex array, or real array with last dim==2 interpreted as [I,Q]
      - .mat: specify mat_key, or auto-pick first non-private variable
    """
    fmt = fmt.lower()

    if fmt == "npy":
        x = np.load(path)
    elif fmt == "mat":
        if sio is None:
            raise RuntimeError("scipy is required to read .mat files. Please `pip install scipy`.")
        md = sio.loadmat(path)
        if mat_key is None:
            keys = [k for k in md.keys() if not k.startswith("__")]
            if not keys:
                raise ValueError(f"No usable variables found in MAT file: {path}")
            mat_key = keys[0]
        x = md[mat_key]
    else:
        raise ValueError(f"Unsupported waveform format '{fmt}'. Use 'npy' or 'mat'.")

    x = np.asarray(x)
    x = np.squeeze(x)

    if np.iscomplexobj(x):
        xc = x.astype(np.complex64)
    else:
        if (x.ndim >= 1) and (x.shape[-1] == 2) and (iq_layout in ("auto", "iq2")):
            i = x[..., 0]
            q = x[..., 1]
            xc = (i + 1j * q).astype(np.complex64)
            xc = np.squeeze(xc)
        else:
            xc = x.astype(np.float32).astype(np.complex64)

    xc = np.reshape(xc, (-1,))
    return xc


def _normalize_rms(x: np.ndarray, target_rms: Optional[float]) -> np.ndarray:
    if target_rms is None:
        return x
    p = np.mean(np.abs(x) ** 2)
    if p <= 0:
        return x
    rms = float(np.sqrt(p))
    if rms == 0:
        return x
    return x * (target_rms / rms)


def _pad_or_trim_1d(x: np.ndarray, n: int, pad_mode: str, trim_mode: str) -> np.ndarray:
    if len(x) == n:
        return x

    if len(x) > n:
        if trim_mode == "first":
            return x[:n]
        elif trim_mode == "center":
            start = (len(x) - n) // 2
            return x[start:start + n]
        elif trim_mode == "last":
            return x[-n:]
        else:
            raise ValueError(f"Unknown trim_mode '{trim_mode}'")
    else:
        if pad_mode == "zero":
            y = np.zeros((n,), dtype=np.complex64)
            y[:len(x)] = x
            return y
        elif pad_mode == "repeat":
            reps = (n + len(x) - 1) // len(x)
            y = np.tile(x, reps)[:n]
            return y.astype(np.complex64)
        else:
            raise ValueError(f"Unknown pad_mode '{pad_mode}'")


def augment_from_config(cfg: Dict[str, Any]) -> np.ndarray:
    """
    Config-driven stochastic TDL channel augmentation.
    Ported from CLIP_datagen/Stochastic/stochastic.py.
    TF/Sionna are imported lazily so the rest of the app works without them.
    """
    try:
        import tensorflow as tf
        from sionna.phy import config as sn_config
        from sionna.phy.channel.time_channel import TimeChannel
        from sionna.phy.channel.tr38901.tdl import TDL
    except ImportError as e:
        raise ImportError(
            "Stochastic TDL augmentation requires TensorFlow and Sionna. "
            "Install them with: pip install tensorflow sionna\n"
            f"Original error: {e}"
        )

    # ---- Seed ----
    seed = cfg.get("seed", None)
    if seed is not None:
        sn_config.seed = int(seed)

    # ---- Waveform (input-side only) ----
    wcfg = cfg["waveform"]
    if "_preloaded_data" in wcfg:
        x = np.asarray(wcfg["_preloaded_data"]).astype(np.complex64).reshape(-1)
    else:
        x = _load_waveform(
            path=wcfg["path"],
            fmt=wcfg.get("format", "npy"),
            mat_key=wcfg.get("mat_key", None),
            iq_layout=wcfg.get("iq_layout", "auto"),
        )
    x = _normalize_rms(x, wcfg.get("normalize_rms", None))

    aug = cfg["augmentation"]
    out_n = int(aug["output_num_samples"])
    pad_mode = aug.get("pad_mode", "repeat")
    trim_mode = aug.get("trim_mode", "first")

    # Choose input length = output length (we'll crop/pad after channel)
    x_in = _pad_or_trim_1d(x, out_n, pad_mode=pad_mode, trim_mode=trim_mode)

    # Inherent waveform sample rate
    waveform_sr = float(wcfg["sample_rate_hz"])

    # ---- Channel (stochastic sim side) ----
    chcfg = cfg["channel"]
    if chcfg.get("type", "").lower() != "tdl":
        raise NotImplementedError("Only TDL is implemented. Extend for CDL/UMi/UMa/RMa later.")

    carrier_frequency = float(chcfg["carrier_frequency_hz"])
    channel_sr = float(chcfg.get("sample_rate_hz", waveform_sr))

    tdl = TDL(
        model=chcfg["profile"],
        delay_spread=float(chcfg["delay_spread_s"]),
        carrier_frequency=carrier_frequency,
        min_speed=float(chcfg.get("min_speed_mps", 0.0)),
        max_speed=chcfg.get("max_speed_mps", None),
        num_rx_ant=int(chcfg.get("num_rx_ant", 1)),
        num_tx_ant=int(chcfg.get("num_tx_ant", 1)),
    )

    time_ch = TimeChannel(
        channel_model=tdl,
        bandwidth=channel_sr,
        num_time_samples=len(x_in),
        maximum_delay_spread=float(chcfg.get("maximum_delay_spread_s", 3e-6)),
        l_min=chcfg.get("l_min", None),
        l_max=chcfg.get("l_max", None),
        normalize_channel=bool(chcfg.get("normalize_channel", False)),
        return_channel=False,
    )

    # ---- Noise ----
    ncfg = cfg.get("noise", {})
    no = None
    if "no" in ncfg and ncfg["no"] is not None:
        no = float(ncfg["no"])
    elif "snr_db" in ncfg and ncfg["snr_db"] is not None:
        snr_db = float(ncfg["snr_db"])
        p_sig = float(np.mean(np.abs(x_in) ** 2))
        snr_lin = 10.0 ** (snr_db / 10.0)
        no = p_sig / snr_lin

    # ---- Apply channel ----
    # Expected shape: [batch, num_tx, num_tx_ant, num_time_samples]
    x_tf = tf.constant(x_in.reshape(1, 1, 1, -1), dtype=tf.complex64)
    no_tf = None if no is None else tf.constant(no, dtype=tf.float32)

    y_tf = time_ch(x_tf, no=no_tf)
    # y_tf shape: (1, 1, num_rx_ant, num_time_samples)

    multi_channel = cfg.get("multi_channel", False)
    if multi_channel:
        # Return all RX antennas: (num_rx_ant, num_time_samples)
        y_mc = y_tf.numpy()[0, 0, :, :].astype(np.complex64)
        # Pad/trim each antenna row to output length
        y_out = np.zeros((y_mc.shape[0], out_n), dtype=np.complex64)
        for i in range(y_mc.shape[0]):
            y_out[i] = _pad_or_trim_1d(y_mc[i], out_n, pad_mode="zero", trim_mode="first")
        return y_out

    y = y_tf.numpy().reshape(-1).astype(np.complex64)

    # ---- Force output length requested ----
    y_out = _pad_or_trim_1d(y, out_n, pad_mode="zero", trim_mode="first")
    return y_out


class SionnaRTAugmentation(AugmentationBlock):
    """Deterministic RT channel augmentation using Sionna's ApplyTimeChannel.

    Consumes precomputed taps from the sionna_widget engine and a config dict
    matching the multiantenna_config.json schema.
    """

    def __init__(self, config: dict, taps: np.ndarray, multi_channel: bool = False):
        self.config = config
        self.taps = taps
        self.multi_channel = multi_channel

    def apply(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        tap_grid = self.config.get("tap_grid") or {}
        tap_fs = tap_grid.get("sampling_frequency", self.config.get("sample_rate"))
        if tap_fs is not None and not np.isclose(float(tap_fs), float(fs), rtol=1e-9, atol=0):
            raise ValueError(f"RT taps were sampled at {tap_fs} Hz, but the waveform uses {fs} Hz. Recompute taps at the waveform rate.")
        import tensorflow as tf
        from sionna.phy.channel import ApplyTimeChannel

        seed = self.config.get("seed")
        if seed is not None:
            from sionna.phy import config as sn_config
            sn_config.seed = int(seed)

        x = np.asarray(signal, dtype=np.complex64).reshape(-1)

        # Normalize to unit power
        p = np.mean(np.abs(x) ** 2)
        if p > 0:
            x = x / np.sqrt(p)

        # Scale by TX power (dBm -> watts -> amplitude)
        tx_power_dbm = float(self.config["transmitters"][0]["power_dbm"])
        p_lin = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
        x = x * np.sqrt(p_lin)

        # Preserve every input sample. A shorter configured length used to
        # silently truncate data and create a length-based class cue.
        requested_T = self.config.get("waveform_length")
        T = len(x) if requested_T is None else max(len(x), int(requested_T))
        if requested_T is not None and int(requested_T) < len(x):
            print(f"Sionna RT requested {requested_T} samples for a {len(x)}-sample waveform; preserving the full waveform.")
        if len(x) < T:
            if "Zero Padded" in str(self.config.get("zero_padding", "Zero Padded")):
                x = np.pad(x, (0, T - len(x)), mode="constant", constant_values=0)
            else:
                tiled = np.tile(x, (T + len(x) - 1) // len(x))
                x = tiled[:T].astype(np.complex64)
        else:
            x = x[:T]

        # Replicate across TX antennas: (1, 1, num_tx_ant, T)
        tx_arr = self.config.get("tx_antenna_array", {})
        num_tx_ant = tx_arr.get("num_rows", 1) * tx_arr.get("num_cols", 1)
        input_wave = np.stack([x] * num_tx_ant, axis=0)
        input_wave = input_wave[np.newaxis, np.newaxis, :, :]

        taps = self.taps
        l_min = int(tap_grid.get("l_min", 0))
        if l_min > 0:
            # ApplyTimeChannel indexes its filter from zero; retain the
            # physical lag when the selected interval starts later.
            taps = np.pad(taps, [(0, 0)] * (taps.ndim - 1) + [(l_min, 0)])
        L_TOT = taps.shape[-1]

        # Noise: dBm -> watts
        noise_power_dbm = float(self.config.get("noise_power_dBm", -108))
        noise_linear = 10.0 ** (noise_power_dbm / 10.0) * 1e-3

        apply_ch = ApplyTimeChannel(num_time_samples=T, l_tot=L_TOT)
        output = apply_ch(
            tf.constant(input_wave, dtype=tf.complex64),
            tf.constant(taps, dtype=tf.complex64),
            tf.constant(noise_linear, dtype=tf.float32),
        )
        # output shape: (1, 1, num_rx_ant, T)
        out_np = output.numpy()
        if self.multi_channel:
            # Return all RX antennas: (num_rx_ant, T)
            return out_np[0, 0, :, :].astype(np.complex64)
        rx_idx = int(self.config.get("rx_antenna_index", 0))
        return out_np[0, 0, rx_idx, :].astype(np.complex64)

    def __repr__(self) -> str:
        scene = self.config.get("filename", "?")
        return f"SionnaRTAugmentation(scene={scene}, taps_shape={self.taps.shape})"


class StochasticTDLAugmentation(AugmentationBlock):
    """
    Config-driven stochastic TDL channel augmentation block.
    Wraps augment_from_config() using the CLIP_datagen config schema.
    """

    def __init__(self, config: dict, multi_channel: bool = False):
        self.config = config
        self.multi_channel = multi_channel

    def apply(self, signal: np.ndarray, fs: float, **kwargs) -> np.ndarray:
        cfg = copy.deepcopy(self.config)
        cfg["waveform"]["_preloaded_data"] = signal
        cfg["multi_channel"] = self.multi_channel
        return augment_from_config(cfg)

    def __repr__(self) -> str:
        profile = self.config.get("channel", {}).get("profile", "?")
        snr = self.config.get("noise", {}).get("snr_db", "?")
        return f"StochasticTDLAugmentation(profile={profile}, snr_db={snr})"

class MeasuredChannelAugmentation(AugmentationBlock):
    """Convolve a signal with a measured channel impulse response.

    The third channel source in the app, beside the stochastic TDL model and
    the Sionna ray tracer: a CIR recorded from real hardware, loaded by
    ``backend.channel_bank``.

    Real passband input stays real.  A measured CIR is generally complex —
    it describes the complex baseband channel — so applying it directly to a
    passband waveform would invent a quadrature component the signal does not
    have.  The analytic signal is used instead, matching how
    ``ScalarAmplitudeAndPhaseShift`` handles the same situation, and the real
    part is taken at the end.  Complex baseband input is convolved directly.
    """

    def __init__(self, channel, normalize: bool = True,
                 snr_db: Optional[float] = None, seed: Optional[int] = None):
        """
        Args:
            channel: a ``channel_bank.MeasuredChannel``
            normalize: scale taps to unit energy so the channel changes the
                shape of the signal without changing its overall power
            snr_db: optional AWGN applied after the channel; None means none
            seed: RNG seed for that noise
        """
        self.channel = channel
        self.normalize = normalize
        self.snr_db = snr_db
        self.seed = seed

    def _taps_at_waveform_rate(self, fs: float) -> np.ndarray:
        """Resample a measured CIR so each tap has the correct physical delay."""
        taps = np.asarray(self.channel.taps, dtype=np.complex128)
        source_fs = self.channel.fs
        self._last_waveform_fs = float(fs)
        self._last_resample_ratio = (1, 1)

        if source_fs is None or np.isclose(source_fs, fs, rtol=1e-9, atol=0.0):
            return taps
        if source_fs <= 0 or fs <= 0:
            raise ValueError("Measured-channel and waveform sample rates must be positive.")

        ratio = Fraction(float(fs) / float(source_fs)).limit_denominator(10_000)
        self._last_resample_ratio = (ratio.numerator, ratio.denominator)
        return resample_poly(taps, ratio.numerator, ratio.denominator)

    def apply(self, signal: np.ndarray, fs: float, **_kwargs) -> np.ndarray:
        taps = self._taps_at_waveform_rate(fs)
        if self.normalize:
            energy = np.sqrt(np.sum(np.abs(taps) ** 2))
            taps = taps / energy if energy > 0 else taps
        if taps.size == 0:
            return signal.copy()

        x = np.asarray(signal)
        if np.iscomplexobj(x):
            y = np.convolve(x, taps)[: len(x)]
        else:
            # Real passband: filter the analytic signal, then come back to real
            # so the output stays the same kind of signal as the input.
            analytic = hilbert(x.astype(np.float64))
            y = np.real(np.convolve(analytic, taps)[: len(x)])
            y = y.astype(np.float32)

        if self.snr_db is not None:
            y = AWGNAugmentation(snr_db=self.snr_db, seed=self.seed).apply(y, fs)
        return y

    def to_config(self) -> dict:
        """Serialisable description, stored in the augmented dataset metadata."""
        return {
            "channel_name": self.channel.name,
            "source_path": self.channel.source_path,
            "num_taps": int(self.channel.num_taps),
            "rms_delay_spread_samples": float(self.channel.delay_spread_samples),
            "channel_fs": self.channel.fs,
            "waveform_fs": getattr(self, "_last_waveform_fs", None),
            "resample_ratio": list(getattr(self, "_last_resample_ratio", (1, 1))),
            "normalize": bool(self.normalize),
            "snr_db": self.snr_db,
            "seed": self.seed,
        }

    def __repr__(self) -> str:
        return (f"MeasuredChannelAugmentation(channel={self.channel.name!r}, "
                f"taps={self.channel.num_taps}, snr_db={self.snr_db})")
