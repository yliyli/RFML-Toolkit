"""
BulkAugmentationThread — applies the currently-configured channel
augmentation to every entry in a source DatasetManager and saves the
results via a destination DatasetManager.

Modeled on DatasetGeneratorThread.  Each file gets its own RNG derived
from (master_seed + entry_index) so runs are reproducible yet per-file
unique.
"""

from __future__ import annotations

import traceback
import copy
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from mixedsignal_gui.backend.augmentation import (
    AugmentationPipeline,
    AWGNAugmentation,
    FrequencyShift,
    ScalarAmplitudeAndPhaseShift,
    StochasticTDLAugmentation,
    SionnaRTAugmentation,
    MeasuredChannelAugmentation,
)
from mixedsignal_gui.backend.parameter_range import ParameterRange
from mixedsignal_gui.backend.experiment_settings import experiment_seed


class BulkAugmentationThread(QThread):
    """Run an augmentation pass over every dataset in *source_manager*."""

    progress = Signal(int, int, str)       # (current_1based, total, entry_name)
    finished = Signal(int, int, str)       # (success_count, error_count, dest_dir)
    error = Signal(str)                    # fatal error message

    def __init__(
        self,
        source_manager,
        dest_manager,
        augmentation_kind: str,          # "awgn" | "stoch_tdl" | "rt" | "measured"
        ranges: dict[str, ParameterRange],
        static_config: dict,
        rt_taps: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
        measured_channels: Optional[list] = None,
    ) -> None:
        super().__init__()
        self._source = source_manager
        self._dest = dest_manager
        self._kind = augmentation_kind
        self._ranges = dict(ranges)
        self._static = copy.deepcopy(static_config)
        self._rt_taps = rt_taps
        self._measured_channels = list(measured_channels) if measured_channels else []
        self._last_channel_name = None
        self._last_channel_index = None
        self._seed = int(seed) if seed is not None else experiment_seed()
        if self._seed < 0:
            raise ValueError("Run seed must be nonnegative.")
        self._cancel = False

    def request_cancel(self) -> None:
        self._cancel = True

    # ------------------------------------------------------------------
    def run(self) -> None:  # noqa: C901 — intentionally flat
        try:
            entries = sorted(self._source.scan(), key=lambda entry: entry["name"])
        except Exception as exc:
            self.error.emit(f"Failed to scan source folder: {exc}")
            return

        total = len(entries)
        if total == 0:
            self.error.emit("Source folder contains no datasets.")
            return

        ok_count = 0
        err_count = 0

        for idx, entry in enumerate(entries):
            if self._cancel:
                break

            name = entry.get("name", f"entry_{idx}")
            self.progress.emit(idx + 1, total, name)

            try:
                signal = self._source.load_signal(entry)
                fs = entry.get("fs", 1.0)

                rng = np.random.default_rng(self._seed + idx)

                # Sample every range parameter for this entry
                sampled: dict[str, float] = {}
                for key, pr in self._ranges.items():
                    sampled[key] = pr.sample(rng)

                self._augmentation_seed = int(rng.integers(0, 2**31))
                augmented = self._apply_one(signal, fs, sampled, rng)

                # For multi-channel output, pick one random antenna
                if augmented.ndim == 2 and augmented.shape[0] > 1:
                    ant_idx = int(rng.integers(0, augmented.shape[0]))
                    augmented_save = augmented[ant_idx]
                else:
                    augmented_save = augmented
                    ant_idx = None

                # Build metadata
                meta = dict(entry)
                meta.pop("_npy_path", None)
                meta.pop("_json_path", None)
                meta["augmented"] = True
                meta["source"] = "bulk_augmented"
                meta["original_name"] = name
                meta["fs"] = fs
                meta["augmentation_type"] = self._kind
                meta["run_seed"] = self._seed
                meta["entry_index"] = idx
                meta["entry_seed"] = self._seed + idx
                meta["augmentation_seed"] = self._augmentation_seed

                aug_cfg: dict = {}
                for key, pr in self._ranges.items():
                    aug_cfg[key] = pr.to_metadata(sampled[key])
                # Include static config entries
                for key, val in self._static.items():
                    if key not in aug_cfg:
                        aug_cfg[key] = copy.deepcopy(val)
                if hasattr(self, "_applied_config"):
                    aug_cfg["applied_config"] = copy.deepcopy(self._applied_config)
                meta["augmentation_config"] = aug_cfg

                if ant_idx is not None:
                    meta["selected_antenna"] = ant_idx
                    meta["total_antennas"] = int(augmented.shape[0])

                # Which measured channel this file was convolved with.  Without
                # it a bank-sampled run is unauditable: every output looks alike
                # in metadata even though each used a different CIR.
                if self._kind == "measured" and self._last_channel_index is not None:
                    meta["measured_channel_index"] = self._last_channel_index
                    meta["measured_channel_name"] = self._last_channel_name
                    meta["measured_bank_size"] = len(self._measured_channels)

                out_name = self._dest._unique_name(f"{name}_aug")
                self._dest.save(out_name, augmented_save, meta)
                ok_count += 1

            except Exception:
                traceback.print_exc()
                err_count += 1

        self.finished.emit(ok_count, err_count, str(self._dest.datasets_dir))

    # ------------------------------------------------------------------
    def _apply_one(
        self,
        signal: np.ndarray,
        fs: float,
        sampled: dict[str, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build and apply the augmentation block for a single entry."""

        if self._kind == "awgn":
            pipeline = AugmentationPipeline()
            if self._static.get("awgn_enabled", True):
                pipeline.add(AWGNAugmentation(snr_db=sampled.get("snr_db", 20.0),
                                              seed=self._augmentation_seed))
            if self._static.get("amp_phase_enabled", False):
                phase_rad = np.deg2rad(sampled.get("phase_deg", 0.0))
                pipeline.add(ScalarAmplitudeAndPhaseShift(
                    amplitude=sampled.get("amplitude", 1.0), phi=phase_rad))
            if self._static.get("freq_shift_enabled", False):
                pipeline.add(FrequencyShift(
                    delta_f=sampled.get("freq_shift_hz", 0.0)))
            return pipeline.apply(signal, fs)

        elif self._kind == "stoch_tdl":
            config = copy.deepcopy(self._static.get("stoch_config", {}))
            # Patch sampled values into the config
            if "delay_spread_ns" in sampled:
                config["channel"]["delay_spread_s"] = sampled["delay_spread_ns"] * 1e-9
            if "stoch_snr_db" in sampled:
                config["noise"]["snr_db"] = sampled["stoch_snr_db"]
            # Fresh seed per entry
            config["seed"] = self._augmentation_seed
            self._applied_config = copy.deepcopy(config)
            multi_ch = self._static.get("multi_channel", False)
            block = StochasticTDLAugmentation(config, multi_channel=multi_ch)
            return block.apply(signal, fs)

        elif self._kind == "rt":
            config = copy.deepcopy(self._static.get("rt_config", {}))
            config["seed"] = self._augmentation_seed
            self._applied_config = copy.deepcopy(config)
            multi_ch = self._static.get("multi_channel", False)
            block = SionnaRTAugmentation(config, self._rt_taps,
                                         multi_channel=multi_ch)
            return block.apply(signal, fs)

        elif self._kind == "measured":
            # Draw a channel per file when the bank holds more than one and the
            # caller asked to randomise.  The index comes from this entry's own
            # generator, so a run over N files uses N independently drawn
            # channels and still reproduces exactly from the master seed --
            # unlike np.random, which the interactive path uses because it only
            # ever augments one file at a time.
            channels = self._measured_channels or []
            if not channels:
                raise ValueError("measured augmentation requires a non-empty channel bank")

            if self._static.get("random_channel", True) and len(channels) > 1:
                idx = int(rng.integers(0, len(channels)))
            else:
                idx = int(self._static.get("channel_index", 0))
                idx = max(0, min(idx, len(channels) - 1))

            channel = channels[idx]
            self._last_channel_name = getattr(channel, "name", str(idx))
            self._last_channel_index = idx

            snr = sampled.get("meas_snr_db") if self._static.get("meas_awgn_enabled") else None
            block = MeasuredChannelAugmentation(
                channel,
                normalize=self._static.get("normalize", True),
                snr_db=snr,
                seed=self._augmentation_seed,
            )
            result = block.apply(signal, fs)
            self._applied_config = block.to_config()
            return result

        else:
            raise ValueError(f"Unknown augmentation kind: {self._kind!r}")
