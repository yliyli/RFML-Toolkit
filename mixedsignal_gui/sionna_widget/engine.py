"""Sionna RT simulation engine for 1 TX + 1 RX.

Supports configurable antenna arrays, PathSolver options, TX power/velocity,
taps computation from ``paths.cir()`` on the waveform grid, and config export matching the
``multiantenna_config.json`` schema used by ``CLIP_datagen/RT/GenCode/runConfigs.py``.
"""

import numpy as np
import tensorflow as tf
from typing import Optional, List, Dict, Any
from mixedsignal_gui.backend.experiment_settings import experiment_seed

from sionna.rt import (
    load_scene, PlanarArray, Transmitter, Receiver, PathSolver,
)

_DEFAULT_ANTENNA_CFG = {
    "num_rows": 1,
    "num_cols": 1,
    "vert_spacing": 0.5,
    "hori_spacing": 0.5,
    "pattern": "iso",
    "polarization": "V",
    "polarization_model": "tr38901_2",
}

_DEFAULT_SOLVER_OPTS = {
    "max_depth": 5,
    "samples_per_src": 1_000_000,
    "synthetic_array": True,
    "line_of_sight": True,
    "specular_reflection": True,
    "diffuse_reflection": True,
    "refraction": True,
}


class SimpleSimulationEngine:
    """Sionna RT engine: one TX, one RX, configurable antenna arrays."""

    TX_NAME = "sionna_tx"
    RX_NAME = "sionna_rx"

    def __init__(self):
        self._scene = None
        self._scene_path: Optional[str] = None
        self._path_solver = None

        # Parameters
        self._frequency: float = 3.5e9

        # Antenna array configs
        self._tx_array_cfg: Dict[str, Any] = dict(_DEFAULT_ANTENNA_CFG)
        self._rx_array_cfg: Dict[str, Any] = dict(_DEFAULT_ANTENNA_CFG)

        # PathSolver options
        self._solver_opts: Dict[str, Any] = dict(_DEFAULT_SOLVER_OPTS)

        # TX power, velocities
        self._tx_power_dbm: float = 30.0
        self._tx_velocity: List[float] = [0.0, 0.0, 0.0]
        self._rx_velocity: List[float] = [0.0, 0.0, 0.0]

        # Positions
        self._tx_position: List[float] = [0.0, 0.0, 10.0]
        self._rx_position: List[float] = [50.0, 0.0, 1.5]

        self._last_paths = None
        self._last_taps: Optional[np.ndarray] = None
        self._tap_grid: Optional[dict] = None

    # -- properties --------------------------------------------------------

    @property
    def scene(self):
        return self._scene

    @property
    def frequency(self) -> float:
        return self._frequency

    @property
    def tx_position(self) -> List[float]:
        return list(self._tx_position)

    @property
    def rx_position(self) -> List[float]:
        return list(self._rx_position)

    @property
    def last_paths(self):
        return self._last_paths

    @property
    def last_taps(self) -> Optional[np.ndarray]:
        return self._last_taps

    # -- scene loading -----------------------------------------------------

    def load_scene(self, path: str) -> bool:
        """Load an XML scene, configure arrays from stored configs, create TX/RX pair."""
        # Clear stale results from any previous scene
        self._last_paths = None
        self._last_taps = None
        self._tap_grid = None

        try:
            self._scene = load_scene(path)
            self._scene_path = path
            self._scene.frequency = self._frequency

            self._rebuild_arrays()
            self._setup_transceivers()
            self._path_solver = PathSolver()
            return True
        except Exception as e:
            import traceback
            print(f"Failed to load scene: {e}")
            traceback.print_exc()
            self._scene = None
            self._scene_path = None
            return False

    def _rebuild_arrays(self):
        """Build and assign PlanarArray objects from stored configs."""
        if self._scene is None:
            return
        self._scene.tx_array = PlanarArray(
            num_rows=self._tx_array_cfg["num_rows"],
            num_cols=self._tx_array_cfg["num_cols"],
            vertical_spacing=self._tx_array_cfg["vert_spacing"],
            horizontal_spacing=self._tx_array_cfg["hori_spacing"],
            pattern=self._tx_array_cfg["pattern"],
            polarization=self._tx_array_cfg["polarization"],
        )
        self._scene.rx_array = PlanarArray(
            num_rows=self._rx_array_cfg["num_rows"],
            num_cols=self._rx_array_cfg["num_cols"],
            vertical_spacing=self._rx_array_cfg["vert_spacing"],
            horizontal_spacing=self._rx_array_cfg["hori_spacing"],
            pattern=self._rx_array_cfg["pattern"],
            polarization=self._rx_array_cfg["polarization"],
        )

    def _setup_transceivers(self):
        """Create (or recreate) the fixed TX/RX pair."""
        for name in (self.TX_NAME, self.RX_NAME):
            try:
                self._scene.remove(name)
            except Exception:
                pass

        tx = Transmitter(
            name=self.TX_NAME,
            position=self._tx_position,
            orientation=[0.0, 0.0, 0.0],
            power_dbm=self._tx_power_dbm,
            velocity=self._tx_velocity,
        )
        self._scene.add(tx)

        rx = Receiver(
            name=self.RX_NAME,
            position=self._rx_position,
            orientation=[0.0, 0.0, 0.0],
            velocity=self._rx_velocity,
        )
        self._scene.add(rx)

    # -- antenna array setters ---------------------------------------------

    def set_antenna_arrays(self, tx_cfg: Dict[str, Any], rx_cfg: Dict[str, Any]):
        """Update antenna array configs and rebuild if scene is loaded."""
        self._tx_array_cfg = {**_DEFAULT_ANTENNA_CFG, **tx_cfg}
        self._rx_array_cfg = {**_DEFAULT_ANTENNA_CFG, **rx_cfg}
        if self._scene is not None:
            self._rebuild_arrays()

    # -- solver options setter ---------------------------------------------

    def set_solver_options(self, opts: Dict[str, Any]):
        """Update PathSolver options."""
        self._solver_opts = {**_DEFAULT_SOLVER_OPTS, **opts}

    # -- parameter setters -------------------------------------------------

    def set_frequency(self, hz: float):
        self._frequency = hz
        if self._scene is not None:
            self._scene.frequency = hz

    def set_max_depth(self, n: int):
        self._solver_opts["max_depth"] = n

    def set_num_samples(self, n: int):
        self._solver_opts["samples_per_src"] = n

    def set_tx_power_dbm(self, dbm: float):
        self._tx_power_dbm = dbm
        if self._scene is not None:
            try:
                obj = self._scene.get(self.TX_NAME)
                if obj is not None:
                    obj.power_dbm = dbm
            except Exception:
                pass

    def set_tx_velocity(self, vel: List[float]):
        self._tx_velocity = list(vel)

    def set_rx_velocity(self, vel: List[float]):
        self._rx_velocity = list(vel)

    def set_tx_position(self, pos: List[float]):
        self._tx_position = list(pos)
        if self._scene is not None:
            try:
                obj = self._scene.get(self.TX_NAME)
                if obj is not None:
                    obj.position = pos
            except Exception:
                pass

    def set_rx_position(self, pos: List[float]):
        self._rx_position = list(pos)
        if self._scene is not None:
            try:
                obj = self._scene.get(self.RX_NAME)
                if obj is not None:
                    obj.position = pos
            except Exception:
                pass

    # -- computation -------------------------------------------------------

    def compute_paths(self) -> Optional[dict]:
        """Run ray tracing and return extracted path data."""
        if self._scene is None or self._path_solver is None:
            return None
        try:
            opts = self._solver_opts
            paths = self._path_solver(
                scene=self._scene,
                max_depth=opts.get("max_depth", 5),
                samples_per_src=opts.get("samples_per_src", 1_000_000),
                synthetic_array=opts.get("synthetic_array", True),
                los=opts.get("line_of_sight", True),
                specular_reflection=opts.get("specular_reflection", True),
                diffuse_reflection=opts.get("diffuse_reflection", True),
                refraction=opts.get("refraction", True),
                seed=experiment_seed(),
            )
            self._last_paths = paths
            return self._extract_path_results(paths)
        except Exception as e:
            import traceback
            print(f"Path computation failed: {e}")
            traceback.print_exc()
            return None

    def compute_taps(self, bandwidth: float, sampling_frequency: float,
                     l_min: int = 0, l_max: int = 200) -> Optional[np.ndarray]:
        """Compute channel taps from the last computed paths.

        Returns a static array with shape (num_rx, num_rx_ant, num_tx,
        num_tx_ant, 1, L_TOT), or None on failure.
        """
        self._last_taps = None
        self._tap_grid = None
        if self._last_paths is None:
            return None
        try:
            if bandwidth <= 0 or sampling_frequency <= 0 or l_max < l_min:
                raise ValueError("Positive bandwidth/sample rate and l_max >= l_min are required.")
            coefficients, delays = self._last_paths.cir(
                sampling_frequency=sampling_frequency,
                normalize_delays=False,
                out_type="numpy",
            )
            # Paths.taps() uses bandwidth, not sampling_frequency, for its
            # delay grid. Evaluate the same band-limited CIR at l/fs instead.
            # BW/fs is the discrete-convolution scaling for oversampled taps.
            delays = np.asarray(delays)
            if delays.ndim == 3:  # synthetic arrays share delays across antennas
                delays = delays[:, None, :, None, :]
            lag_times = np.arange(l_min, l_max + 1) / sampling_frequency
            kernel = (bandwidth / sampling_frequency) * np.sinc(
                bandwidth * (lag_times - delays[..., None, None]))
            taps = np.sum(np.asarray(coefficients)[..., None] * kernel, axis=4).astype(np.complex64)
            self._last_taps = taps
            self._tap_grid = {
                "sampling_frequency": float(sampling_frequency),
                "bandwidth": float(bandwidth), "l_min": int(l_min), "l_max": int(l_max),
            }
            return taps
        except Exception as e:
            import traceback
            print(f"Taps computation failed: {e}")
            traceback.print_exc()
            return None

    # -- config export -----------------------------------------------------

    def get_full_config(self) -> Dict[str, Any]:
        """Export engine state as a dict matching multiantenna_config.json schema."""
        return {
            "cpu_mode": True,
            "tap_grid": dict(self._tap_grid) if self._tap_grid is not None else None,
            "bandwidth": self._tap_grid["bandwidth"] if self._tap_grid else (
                self._scene.bandwidth if self._scene and hasattr(self._scene, "bandwidth") else 5e6),
            "center_frequency": self._frequency,
            "filename": self._scene_path or "",
            "waveform_length": 1024,
            "zero_padding": "Zero Padded",
            "noise_power_dBm": -108,
            "temperature": "undefined",
            "seed": experiment_seed(),
            "sample_rate": self._tap_grid["sampling_frequency"] if self._tap_grid else 30.72e6,
            "transmitters": [
                {
                    "name": self.TX_NAME,
                    "power_dbm": self._tx_power_dbm,
                    "position": list(self._tx_position),
                    "orientation": [0.0, 0.0, 0.0],
                    "look_at": list(self._rx_position),
                    "velocity": list(self._tx_velocity),
                    "waveform_path": "",
                }
            ],
            "receivers": [
                {
                    "name": self.RX_NAME,
                    "position": list(self._rx_position),
                    "orientation": [0.0, 0.0, 0.0],
                    "look_at": ["undefined"],
                    "velocity": list(self._rx_velocity),
                }
            ],
            "tx_antenna_array": dict(self._tx_array_cfg),
            "rx_antenna_array": dict(self._rx_array_cfg),
            "path_solver": {
                "max_depth": self._solver_opts.get("max_depth", 5),
                "max_num_paths_per_src": self._solver_opts.get("samples_per_src", 1_000_000),
                "samples_per_src": self._solver_opts.get("samples_per_src", 1_000_000),
                "synthetic_array": self._solver_opts.get("synthetic_array", True),
                "line_of_sight": self._solver_opts.get("line_of_sight", True),
                "specular_reflection": self._solver_opts.get("specular_reflection", True),
                "diffuse_reflection": self._solver_opts.get("diffuse_reflection", True),
                "refraction": self._solver_opts.get("refraction", True),
            },
        }

    @staticmethod
    def _to_numpy(tensor) -> np.ndarray:
        """Convert a Sionna/drjit/TF tensor to a numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
        return np.array(tensor)

    @staticmethod
    def _squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
        """Squeeze singleton dims between axis-0 (depth) and last two (paths, 3)."""
        while arr.ndim > 3:
            squeezed = False
            for ax in range(1, arr.ndim - 2):
                if arr.shape[ax] == 1:
                    arr = arr.squeeze(axis=ax)
                    squeezed = True
                    break
            if not squeezed:
                break
        return arr

    def _build_path_polylines(self, paths, power_mask) -> list:
        """Build a list of Nx3 polylines from Sionna path data.

        Uses ``paths.valid`` and ``paths.interactions`` to determine which
        depth slots contain real interaction points, ``paths.sources`` /
        ``paths.targets`` for accurate start/end positions, and the
        *power_mask* to only include paths with non-negligible power.

        Returns a list of np.ndarray, each shaped (num_points, 3).
        """
        try:
            verts = paths.vertices
            if verts is None:
                return []
            v_np = self._squeeze_to_3d(self._to_numpy(verts))
            if v_np.ndim != 3:
                return []

            max_depth_v, n_paths, _ = v_np.shape

            # Interaction types: [max_depth, ..., num_paths] -> [max_depth, num_paths]
            interactions = None
            if hasattr(paths, "interactions") and paths.interactions is not None:
                inter_np = self._to_numpy(paths.interactions)
                # Squeeze singleton dims, keeping depth (axis 0) and paths (last)
                while inter_np.ndim > 2:
                    for ax in range(1, inter_np.ndim - 1):
                        if inter_np.shape[ax] == 1:
                            inter_np = inter_np.squeeze(axis=ax)
                            break
                    else:
                        break
                if inter_np.ndim == 2 and inter_np.shape[1] == n_paths:
                    interactions = inter_np  # [max_depth, num_paths]

            # Valid mask: [..., num_paths] -> [num_paths]
            valid_np = None
            if hasattr(paths, "valid") and paths.valid is not None:
                valid_np = self._to_numpy(paths.valid).flatten().astype(bool)

            # Source/target positions (actual TX/RX as used by Sionna)
            src = np.array(self._tx_position, dtype=np.float64)
            tgt = np.array(self._rx_position, dtype=np.float64)
            if hasattr(paths, "sources") and paths.sources is not None:
                s_np = self._to_numpy(paths.sources)
                if s_np.size >= 3:
                    src = s_np.flatten()[:3]
            if hasattr(paths, "targets") and paths.targets is not None:
                t_np = self._to_numpy(paths.targets)
                if t_np.size >= 3:
                    tgt = t_np.flatten()[:3]

            # Build polylines only for valid, non-zero-power paths
            # Cap the number of drawn paths for performance
            MAX_DRAW = 500
            path_list = []

            for p in range(n_paths):
                # Skip invalid paths
                if valid_np is not None and not valid_np[p]:
                    continue
                # Skip zero-power paths (use the same mask as delays/gains)
                if p < len(power_mask) and not power_mask[p]:
                    continue

                points = [src.copy()]
                for d in range(max_depth_v):
                    # Check if this depth has a real interaction
                    if interactions is not None:
                        if interactions[d, p] == 0:  # InteractionType.NONE
                            continue
                    pt = v_np[d, p, :]
                    points.append(pt.copy())
                points.append(tgt.copy())

                # Sanity: only keep if we have at least 2 points
                if len(points) >= 2:
                    path_list.append(np.array(points, dtype=np.float64))

                if len(path_list) >= MAX_DRAW:
                    break

            return path_list
        except Exception as e:
            import traceback
            print(f"Warning: could not build path polylines: {e}")
            traceback.print_exc()
            return []

    def _extract_path_results(self, paths) -> dict:
        """Pull delays, complex gains, powers, angles, and vertices.

        With multi-element antenna arrays, ``paths.a`` includes antenna
        dimensions that ``paths.tau`` does not.  We reduce gains to
        per-path quantities using the flat-size ratio so the power mask
        aligns with the delay array regardless of the exact tensor layout.
        """
        results = {}
        try:
            delays = self._to_numpy(paths.tau).flatten()
            n_paths = len(delays)

            # --- complex gains (may include antenna dims) ---
            a_raw = paths.a
            if isinstance(a_raw, tuple):
                a_real = self._to_numpy(a_raw[0])
                a_imag = self._to_numpy(a_raw[1])
                a_flat = (a_real.flatten() + 1j * a_imag.flatten()).astype(np.complex128)
            else:
                a_flat = self._to_numpy(a_raw).flatten().astype(np.complex128)

            # Reduce antenna dims: ratio tells us how many antenna elements
            # each path has been expanded across (e.g. 36 for 6×6 arrays).
            ratio = len(a_flat) // n_paths if n_paths > 0 else 1
            if ratio > 1:
                a_2d = a_flat.reshape(ratio, n_paths)
                per_path_power = (np.abs(a_2d) ** 2).sum(axis=0)
                complex_gains = a_2d.mean(axis=0)
            else:
                per_path_power = np.abs(a_flat) ** 2
                complex_gains = a_flat

            # Filter out zero-power paths
            mask = per_path_power > 1e-30
            results["delays"] = delays[mask]
            results["complex_gains"] = complex_gains[mask]
            results["powers"] = per_path_power[mask]

            # Path types
            if hasattr(paths, "types"):
                results["types"] = self._to_numpy(paths.types)

            # Angle data (departure / arrival)
            for attr, key in [
                ("theta_t", "aod_theta"), ("phi_t", "aod_phi"),
                ("theta_r", "aoa_theta"), ("phi_r", "aoa_phi"),
            ]:
                if hasattr(paths, attr):
                    val = getattr(paths, attr)
                    if val is not None:
                        arr = self._to_numpy(val).flatten()
                        results[key] = arr[mask] if len(arr) == len(mask) else arr

            # Vertex extraction for 3D ray path visualization
            results["vertices"] = self._build_path_polylines(paths, mask)
        except Exception as e:
            import traceback
            print(f"Error extracting path results: {e}")
            traceback.print_exc()
        return results
