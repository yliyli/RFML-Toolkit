"""SionnaWidget - the main drop-in composite QWidget.

Assembles the control panel, 3D viewport, and pipeline visualization panel
into a single embeddable widget with a public API for programmatic control.

When ``viewport_only=True``, only the 3D viewport is created (no control
panel, no pipeline visualization panel).  External code drives the engine
via proxy methods such as ``set_antenna_arrays``, ``set_solver_options``,
``set_channel_params``, etc.

Note: Sionna's drjit backend requires all computation to run on the main
thread (JIT scope IDs cannot cross thread boundaries), so path computation
runs synchronously with a busy cursor.
"""

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QSplitter, QFileDialog, QHBoxLayout, QVBoxLayout, QApplication,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from mixedsignal_gui.sionna_widget.engine import SimpleSimulationEngine
from mixedsignal_gui.sionna_widget.augmentation import SionnaChannelAugmentation, ChannelParameters
from mixedsignal_gui.sionna_widget.controls import SimpleControlPanel
from mixedsignal_gui.sionna_widget.viewport import SimpleViewport
from mixedsignal_gui.sionna_widget.visualization import PipelineVisualizationPanel

from typing import Optional, Dict, Any


class SionnaWidget(QWidget):
    """Drop-in Sionna RT channel widget with configurable antenna arrays.

    Embed in any PySide6 app::

        layout.addWidget(SionnaWidget())                  # full 3-panel
        layout.addWidget(SionnaWidget(viewport_only=True)) # viewport only
    """

    # -- Public signals ----------------------------------------------------
    paths_computed = Signal(object)      # ChannelParameters
    augmentation_ready = Signal(object)  # SionnaChannelAugmentation
    taps_computed = Signal(object)       # np.ndarray
    scene_loaded = Signal(str)           # file path
    error_occurred = Signal(str)

    # Viewport placement signals (forwarded from viewport, useful in viewport_only mode)
    tx_placed = Signal(list)
    rx_placed = Signal(list)

    def __init__(self, parent=None, *, viewport_only: bool = False):
        super().__init__(parent)

        self._viewport_only = viewport_only

        # -- Internal state ------------------------------------------------
        self._engine = SimpleSimulationEngine()
        self._augmentation = SionnaChannelAugmentation()
        self._channel_params: ChannelParameters | None = None
        self._channel_params_override: Dict[str, Any] | None = None
        self._computing = False

        # -- UI ------------------------------------------------------------
        self._controls: SimpleControlPanel | None = None
        self._viz_panel: PipelineVisualizationPanel | None = None

        self._build_ui()
        self._connect_signals()

        # Push default positions into viewport
        self._sync_viewport_markers()

    # -- UI construction ---------------------------------------------------

    def _build_ui(self):
        if self._viewport_only:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            self._viewport = SimpleViewport()
            layout.addWidget(self._viewport)
        else:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)

            splitter = QSplitter(Qt.Horizontal)

            self._controls = SimpleControlPanel()
            self._viewport = SimpleViewport()
            self._viz_panel = PipelineVisualizationPanel()

            splitter.addWidget(self._controls)
            splitter.addWidget(self._viewport)
            splitter.addWidget(self._viz_panel)
            splitter.setSizes([240, 600, 360])

            layout.addWidget(splitter)

    def _connect_signals(self):
        # Viewport placement results — always connected
        self._viewport.tx_placed.connect(self._on_tx_placed)
        self._viewport.rx_placed.connect(self._on_rx_placed)

        if self._controls is not None:
            cp = self._controls

            # Config changes -> push to engine
            cp.config_changed.connect(self._on_config_changed)

            # TX/RX position changes from control panel
            cp.tx_position_changed.connect(self._on_tx_position_changed)
            cp.rx_position_changed.connect(self._on_rx_position_changed)

            # Scene loading
            cp.load_scene_requested.connect(self._on_load_scene)

            # Compute
            cp.compute_requested.connect(self._on_compute_requested)

            # Placement mode
            cp.place_tx_requested.connect(lambda: self._viewport.enter_placement_mode("tx"))
            cp.place_rx_requested.connect(lambda: self._viewport.enter_placement_mode("rx"))

            # Viewport placement mode ended (e.g. Escape) -> uncheck buttons
            self._viewport.gl_widget.placement_mode_changed.connect(
                self._on_placement_mode_changed
            )

            # Antenna and solver config changes
            cp.antenna_config_changed.connect(self._on_antenna_config_changed)
            cp.solver_options_changed.connect(self._on_solver_options_changed)

    # -- Internal slots ----------------------------------------------------

    @Slot()
    def _on_config_changed(self):
        self._engine.set_frequency(self._controls.get_frequency_hz())
        self._engine.set_max_depth(self._controls.get_max_depth())
        self._engine.set_num_samples(self._controls.get_num_samples())

    @Slot(list)
    def _on_tx_position_changed(self, pos):
        self._engine.set_tx_position(pos)
        self._sync_viewport_markers()

    @Slot(list)
    def _on_rx_position_changed(self, pos):
        self._engine.set_rx_position(pos)
        self._sync_viewport_markers()

    @Slot()
    def _on_load_scene(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Sionna Scene", "",
            "XML Files (*.xml);;All Files (*)",
        )
        if path:
            self.load_scene(path)

    @Slot()
    def _on_compute_requested(self):
        self.compute_paths()

    @Slot(list)
    def _on_tx_placed(self, pos):
        self._engine.set_tx_position(pos)
        if self._controls is not None:
            self._controls.set_tx_position(pos)
        self._sync_viewport_markers()
        self._viewport.exit_placement_mode()
        if self._controls is not None:
            self._controls.uncheck_placement_buttons()
        # Forward to public signal for external listeners
        self.tx_placed.emit(pos)

    @Slot(list)
    def _on_rx_placed(self, pos):
        self._engine.set_rx_position(pos)
        if self._controls is not None:
            self._controls.set_rx_position(pos)
        self._sync_viewport_markers()
        self._viewport.exit_placement_mode()
        if self._controls is not None:
            self._controls.uncheck_placement_buttons()
        # Forward to public signal for external listeners
        self.rx_placed.emit(pos)

    @Slot(bool)
    def _on_placement_mode_changed(self, active):
        if not active and self._controls is not None:
            self._controls.uncheck_placement_buttons()

    @Slot()
    def _on_antenna_config_changed(self):
        self._engine.set_antenna_arrays(
            self._controls.get_tx_antenna_config(),
            self._controls.get_rx_antenna_config(),
        )

    @Slot()
    def _on_solver_options_changed(self):
        self._engine.set_solver_options(self._controls.get_solver_options())

    def _on_paths_ready(self, results):
        params = ChannelParameters(
            delays=results.get("delays", np.array([])),
            complex_gains=results.get("complex_gains", np.array([], dtype=np.complex128)),
            powers_linear=results.get("powers", np.array([])),
            aod_theta=results.get("aod_theta"),
            aod_phi=results.get("aod_phi"),
            aoa_theta=results.get("aoa_theta"),
            aoa_phi=results.get("aoa_phi"),
            frequency_hz=self._engine.frequency,
            num_paths=len(results.get("delays", [])),
        )
        self._channel_params = params
        self._augmentation.update_params(params)

        # Draw ray paths in viewport
        vertices = results.get("vertices")
        self._viewport.draw_paths(vertices)

        # Emit public signals
        self.paths_computed.emit(params)
        self.augmentation_ready.emit(self._augmentation)

        # Auto-compute taps after path computation
        self.compute_taps()

    # -- Helpers -----------------------------------------------------------

    def _sync_viewport_markers(self):
        try:
            self._viewport.update_tx_rx(
                self._engine.tx_position,
                self._engine.rx_position,
            )
        except Exception:
            pass  # viewport may not be fully initialized yet

    def _get_channel_params_dict(self) -> Dict[str, Any]:
        """Return channel params from controls or override dict."""
        if self._controls is not None:
            return self._controls.get_channel_params()
        if self._channel_params_override is not None:
            return dict(self._channel_params_override)
        # Sensible defaults when no controls and no override
        return {
            "bandwidth_hz": 5e6,
            "l_min": 0,
            "l_max": 200,
            "tx_power_dbm": 30.0,
            "noise_power_dbm": -108.0,
            "waveform_length": 1024,
            "rx_antenna_index": 0,
        }

    def _get_solver_options_dict(self) -> Dict[str, Any]:
        """Return solver options from controls or engine defaults."""
        if self._controls is not None:
            return self._controls.get_solver_options()
        return {
            "line_of_sight": self._engine._solver_opts.get("line_of_sight", True),
            "specular_reflection": self._engine._solver_opts.get("specular_reflection", True),
            "diffuse_reflection": self._engine._solver_opts.get("diffuse_reflection", True),
            "refraction": self._engine._solver_opts.get("refraction", True),
            "synthetic_array": self._engine._solver_opts.get("synthetic_array", True),
        }

    # -- Public API --------------------------------------------------------

    def get_augmentation_block(self) -> SionnaChannelAugmentation:
        """Return the channel augmentation block (always the same instance)."""
        return self._augmentation

    def get_channel_parameters(self) -> ChannelParameters | None:
        """Return the latest channel parameters, or None if not yet computed."""
        return self._channel_params

    def get_last_taps(self) -> Optional[np.ndarray]:
        """Return the latest computed taps, or None."""
        return self._engine.last_taps

    def compute_taps(self, sampling_frequency: Optional[float] = None) -> Optional[np.ndarray]:
        """Compute channel taps using current control/override values.

        Args:
            sampling_frequency: Explicit sampling frequency in Hz.  When None,
                uses the dataset sample_rate_hz override when available,
                otherwise falls back to bandwidth for standalone use.

        Returns the taps array, or None on failure.
        """
        ch = self._get_channel_params_dict()
        sf = sampling_frequency if sampling_frequency is not None else ch.get("sample_rate_hz", ch["bandwidth_hz"])
        taps = self._engine.compute_taps(
            bandwidth=ch["bandwidth_hz"],
            sampling_frequency=sf,
            l_min=ch["l_min"],
            l_max=ch["l_max"],
        )
        if taps is not None:
            self.taps_computed.emit(taps)
        return taps

    def get_rt_config(self) -> Dict[str, Any]:
        """Assemble full multiantenna_config.json-schema config from engine + controls/override."""
        config = self._engine.get_full_config()
        ch = self._get_channel_params_dict()

        # Keep the selected waveform sample rate distinct from bandwidth.
        config["bandwidth"] = ch["bandwidth_hz"]
        if "sample_rate_hz" in ch:
            config["sample_rate"] = ch["sample_rate_hz"]
        config["noise_power_dBm"] = ch["noise_power_dbm"]
        config["waveform_length"] = ch["waveform_length"]
        config["transmitters"][0]["power_dbm"] = ch["tx_power_dbm"]

        # Waveform padding mode
        if "zero_padding" in ch:
            config["zero_padding"] = ch["zero_padding"]

        # Store rx_antenna_index for augmentation use
        config["rx_antenna_index"] = ch["rx_antenna_index"]

        # Overlay solver options
        solver_opts = self._get_solver_options_dict()
        config["path_solver"]["line_of_sight"] = solver_opts["line_of_sight"]
        config["path_solver"]["specular_reflection"] = solver_opts["specular_reflection"]
        config["path_solver"]["diffuse_reflection"] = solver_opts["diffuse_reflection"]
        config["path_solver"]["refraction"] = solver_opts["refraction"]
        config["path_solver"]["synthetic_array"] = solver_opts["synthetic_array"]

        return config

    def load_scene(self, path: str):
        """Programmatically load a scene file."""
        # Reset computation state so a stuck _computing flag from a prior
        # error does not block future compute_paths() calls.
        self._computing = False

        # Clear existing paths in the viewport (best-effort)
        try:
            self._viewport.gl_widget.clear_paths()
        except Exception as e:
            print(f"Warning: could not clear viewport paths: {e}")

        # macOS Qt can crash in native cursor image registration when this
        # is called from the scene combo box. Do not install a busy cursor.
        ok = self._engine.load_scene(path)

        if ok:
            if self._controls is not None:
                self._controls.set_scene_label(path.rsplit("/", 1)[-1])
            try:
                self._viewport.refresh_scene_meshes(self._engine.scene)
            except Exception as e:
                print(f"Warning: could not refresh viewport meshes: {e}")
            self._sync_viewport_markers()
            self.scene_loaded.emit(path)
        else:
            self.error_occurred.emit(f"Failed to load scene: {path}")

    # -- Proxy methods for external control (viewport_only mode) -----------

    def set_frequency_ghz(self, f: float):
        """Set carrier frequency in GHz."""
        if self._controls is not None:
            self._controls.freq_spin.setValue(f)
        self._engine.set_frequency(f * 1e9)

    def set_tx_position(self, pos: list):
        """Set transmitter position [x, y, z]."""
        self._engine.set_tx_position(pos)
        if self._controls is not None:
            self._controls.set_tx_position(pos)
        self._sync_viewport_markers()

    def set_rx_position(self, pos: list):
        """Set receiver position [x, y, z]."""
        self._engine.set_rx_position(pos)
        if self._controls is not None:
            self._controls.set_rx_position(pos)
        self._sync_viewport_markers()

    def set_max_depth(self, n: int):
        """Set max ray depth."""
        self._engine.set_max_depth(n)

    def set_num_samples(self, n: int):
        """Set number of ray samples."""
        self._engine.set_num_samples(n)

    def set_tx_power_dbm(self, dbm: float):
        """Set TX power in dBm."""
        self._engine.set_tx_power_dbm(dbm)

    def set_antenna_arrays(self, tx_cfg: dict, rx_cfg: dict):
        """Set antenna array configs and rebuild arrays if scene is loaded."""
        self._engine.set_antenna_arrays(tx_cfg, rx_cfg)

    def set_solver_options(self, opts: dict):
        """Set PathSolver options."""
        self._engine.set_solver_options(opts)

    def set_channel_params(self, params_dict: dict):
        """Store channel parameters override (used when _controls is None).

        Expected keys: bandwidth_hz, optional sample_rate_hz, l_min, l_max,
        tx_power_dbm, noise_power_dbm, waveform_length, rx_antenna_index.
        """
        self._channel_params_override = dict(params_dict)

    def enter_placement_mode(self, role: str):
        """Enter placement mode ('tx' or 'rx') in the viewport."""
        self._viewport.enter_placement_mode(role)

    def exit_placement_mode(self):
        """Exit placement mode in the viewport."""
        self._viewport.exit_placement_mode()

    def compute_paths(self):
        """Run path computation on the main thread (drjit requirement).

        Shows computation state in the controls. Emits ``paths_computed`` and
        ``augmentation_ready`` on success, or ``error_occurred`` on failure.
        """
        if self._computing:
            return
        if self._engine.scene is None:
            self.error_occurred.emit("No scene loaded")
            return

        self._computing = True
        if self._controls is not None:
            self._controls.compute_btn.setEnabled(False)
            self._controls.compute_btn.setText("Computing...")
        QApplication.processEvents()

        # Use a single-shot timer to let the UI update before blocking
        QTimer.singleShot(0, self._do_compute)

    def _do_compute(self):
        """Actual computation, deferred via QTimer so the UI can repaint."""
        try:
            result = self._engine.compute_paths()
            if result:
                self._on_paths_ready(result)
            else:
                self.error_occurred.emit("Path computation returned no results")
        except Exception as e:
            self.error_occurred.emit(f"Path computation failed: {e}")
        finally:
            self._computing = False
            if self._controls is not None:
                self._controls.compute_btn.setEnabled(True)
                self._controls.compute_btn.setText("Compute Paths")

    def visualize_pipeline(self, stage_names: list, stage_signals: list,
                           fs: float):
        """Display pipeline visualization for the given stages."""
        if self._viz_panel is not None:
            self._viz_panel.display_pipeline_results(stage_names, stage_signals, fs)
