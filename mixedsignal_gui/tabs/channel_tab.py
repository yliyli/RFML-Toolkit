from datetime import datetime
import copy

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog,
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem,
                               QHeaderView, QSlider, QGridLayout, QComboBox,
                               QDoubleSpinBox, QSpinBox, QStackedWidget, QMessageBox,
                               QButtonGroup, QCheckBox, QScrollArea, QProgressDialog,
                               QListWidget, QInputDialog)
from PySide6.QtCore import Qt, QSettings

from PySide6.QtGui import QPalette, QColor

from mixedsignal_gui.widgets.toggle_switch import ToggleSwitch
from mixedsignal_gui.widgets.noise_spectrum import NoiseSpectrumWidget
from mixedsignal_gui.widgets.waveform_plots import PlottingWidget
from mixedsignal_gui.widgets.wheel_filter import install_wheel_blocker
from mixedsignal_gui.widgets.range_spinbox import RangeSpinBox
from mixedsignal_gui.backend.augmentation import (AugmentationPipeline, AWGNAugmentation,
                                  ScalarAmplitudeAndPhaseShift, FrequencyShift,
                                  StochasticTDLAugmentation, SionnaRTAugmentation,
                                  MeasuredChannelAugmentation)
from mixedsignal_gui.backend.parameter_range import ParameterRange
from mixedsignal_gui.backend.experiment_settings import experiment_seed
from mixedsignal_gui.backend.bulk_augmentation import BulkAugmentationThread
from mixedsignal_gui.backend.dataset_manager import DatasetManager
from mixedsignal_gui.backend.channel_bank import ChannelBank, BIN_DTYPES, DOMAINS
from mixedsignal_gui.sionna_widget.scenes import available_scenes
import numpy as np
import json
import os
from pathlib import Path


class ChannelNoiseTab(QWidget):
    """Channel configuration and noise settings tab"""

    def __init__(self, dataset_manager):
        super().__init__()
        self.dataset_manager = dataset_manager

        # AWGN params
        self.snr_db = 20.0
        self.amplitude = 1.0
        self.phase_deg = 0.0
        self.freq_shift_hz = 0.0
        self.awgn_enabled = True
        self.amp_phase_enabled = False
        self.freq_shift_enabled = False

        # Stochastic TDL params
        self.active_subtab = 0  # 0=AWGN, 1=Stochastic, 2=RT
        self.tdl_profile = "A"
        self.delay_spread_ns = 100.0
        self.stoch_snr_db = 20.0
        self.output_num_samples = 1024

        # RT state
        self.rt_last_taps = None
        self.rt_tap_grid = None
        self.sionna_widget = None  # lazy-created

        # Measured-channel state: CIRs imported from real recordings
        self.channel_bank = ChannelBank()

        self._active_entry = None   # currently selected metadata dict

        self.setup_ui()
        install_wheel_blocker(self)
        self.refresh_dataset_list()

    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left panel - Channel Configuration
        left_panel = self.create_controls_panel()
        layout.addWidget(left_panel, 1)

        # Right panel - Visualization
        right_panel = self.create_visualization_panel()
        layout.addWidget(right_panel, 2)

    def create_controls_panel(self):
        panel = QFrame()
        panel.setObjectName("card")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # Title
        title = QLabel("Channel Augmentation")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Apply channel effects and noise")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Dataset Selection
        dataset_selection = QVBoxLayout()
        dataset_selection.setSpacing(4)
        dataset_label = QLabel("Select Dataset")
        dataset_label.setProperty("class", "section-title")
        dataset_selection.addWidget(dataset_label)

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)
        dataset_selection.addWidget(self.dataset_combo)

        dataset_buttons = QHBoxLayout()
        dataset_buttons.setSpacing(6)
        upload_btn = QPushButton("Upload Dataset")
        upload_btn.clicked.connect(self.upload_dataset)
        dataset_buttons.addWidget(upload_btn)

        refresh_btn = QPushButton("Refresh Datasets")
        refresh_btn.clicked.connect(self.refresh_dataset_list)
        dataset_buttons.addWidget(refresh_btn)
        dataset_selection.addLayout(dataset_buttons)

        layout.addLayout(dataset_selection)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # --- Subtab pills ---
        pill_layout = QHBoxLayout()
        pill_layout.setSpacing(8)

        self.subtab_group = QButtonGroup(self)
        self.subtab_group.setExclusive(True)

        self.awgn_pill = QPushButton("AWGN")
        self.awgn_pill.setObjectName("subtabPill")
        self.awgn_pill.setCheckable(True)
        self.awgn_pill.setChecked(True)

        self.stoch_pill = QPushButton("Stochastic Channel")
        self.stoch_pill.setObjectName("subtabPill")
        self.stoch_pill.setCheckable(True)

        self.rt_pill = QPushButton("Ray Tracing")
        self.rt_pill.setObjectName("subtabPill")
        self.rt_pill.setCheckable(True)

        self.measured_pill = QPushButton("Measured Channel")
        self.measured_pill.setObjectName("subtabPill")
        self.measured_pill.setCheckable(True)

        self.subtab_group.addButton(self.awgn_pill, 0)
        self.subtab_group.addButton(self.stoch_pill, 1)
        self.subtab_group.addButton(self.rt_pill, 2)
        self.subtab_group.addButton(self.measured_pill, 3)

        pill_layout.addWidget(self.awgn_pill)
        pill_layout.addWidget(self.stoch_pill)
        pill_layout.addWidget(self.rt_pill)
        pill_layout.addWidget(self.measured_pill)
        pill_layout.addStretch()
        layout.addLayout(pill_layout)

        # --- Stacked widget ---
        self.subtab_stack = QStackedWidget()
        self.subtab_stack.addWidget(self._create_awgn_page())
        self.subtab_stack.addWidget(self._create_stochastic_page())
        self.subtab_stack.addWidget(self._create_rt_page())
        self.subtab_stack.addWidget(self._create_measured_page())
        layout.addWidget(self.subtab_stack, 1)  # stretch factor 1 → fills all space above buttons

        self.subtab_group.idClicked.connect(self._on_subtab_changed)

        # RT-only: Compute Paths button + status (hidden for AWGN/Stochastic)
        self.rt_compute_btn = QPushButton("Compute Paths")
        self.rt_compute_btn.setObjectName("primaryButton")
        self.rt_compute_btn.clicked.connect(self._rt_compute_paths)
        self.rt_compute_btn.setVisible(False)
        layout.addWidget(self.rt_compute_btn)

        self.rt_status_label = QLabel("No scene loaded")
        self.rt_status_label.setWordWrap(True)
        self.rt_status_label.setProperty("class", "section-subtitle")
        self.rt_status_label.setVisible(False)
        layout.addWidget(self.rt_status_label)

        # Apply Augmentation Button
        apply_btn = QPushButton("Apply Augmentations")
        apply_btn.setObjectName("primaryButton")
        apply_btn.clicked.connect(self.apply_augmentations)
        layout.addWidget(apply_btn)

        self.save_augmented_btn = QPushButton("Save Augmented Dataset")
        self.save_augmented_btn.setEnabled(False)
        self.save_augmented_btn.clicked.connect(self.save_augmented_dataset)
        layout.addWidget(self.save_augmented_btn)

        # Bulk Apply Button
        bulk_btn = QPushButton("Apply to a Dataset Folder")
        bulk_btn.clicked.connect(self.apply_to_all_in_folder)
        layout.addWidget(bulk_btn)

        return panel

    @staticmethod
    def _make_scroll_page():
        """Create a scroll-area-wrapped page with white background matching QFrame#card.

        Returns (page, layout) where *page* goes into the QStackedWidget and
        *layout* is the QVBoxLayout to add controls into.
        """
        _white = QColor(255, 255, 255)
        _dark = QColor(0x1f, 0x29, 0x37)  # #1f2937 — matches stylesheet text

        def _apply_palette(w):
            pal = w.palette()
            pal.setColor(QPalette.Window, _white)
            pal.setColor(QPalette.Base, _white)
            pal.setColor(QPalette.WindowText, _dark)
            pal.setColor(QPalette.Text, _dark)
            w.setPalette(pal)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        _apply_palette(scroll)
        _apply_palette(container)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        scroll.setWidget(container)
        # Re-apply after setWidget since Qt may recreate the viewport
        _apply_palette(scroll.viewport())

        page_layout.addWidget(scroll)
        return page, layout

    # ---- AWGN page ----
    def _create_awgn_page(self):
        page, layout = self._make_scroll_page()

        # AWGN Toggle
        awgn_header = QHBoxLayout()
        awgn_left = QVBoxLayout()
        awgn_title = QLabel("AWGN (Additive White Gaussian Noise)")
        awgn_title.setProperty("class", "section-title")
        awgn_desc = QLabel("Add white Gaussian noise")
        awgn_desc.setProperty("class", "section-subtitle")
        awgn_left.addWidget(awgn_title)
        awgn_left.addWidget(awgn_desc)
        awgn_header.addLayout(awgn_left)
        awgn_header.addStretch()
        self.awgn_toggle = ToggleSwitch()
        self.awgn_toggle.setChecked(self.awgn_enabled)
        awgn_header.addWidget(self.awgn_toggle)
        layout.addLayout(awgn_header)

        # SNR as a RangeSpinBox (from the bulk-augmentation work) kept inside the
        # grey-out wrapper this tab already used, so the control both supports a
        # per-file swept range and still visibly disables with the toggle.
        self.awgn_controls = QWidget()
        awgn_ctrl_layout = QVBoxLayout(self.awgn_controls)
        awgn_ctrl_layout.setContentsMargins(0, 0, 0, 0)

        self.snr_spin = RangeSpinBox()
        snr_header = QHBoxLayout()
        snr_header.addWidget(QLabel("SNR (dB)"))
        snr_header.addStretch()
        snr_header.addWidget(self.snr_spin.vary_checkbox)
        awgn_ctrl_layout.addLayout(snr_header)
        self.snr_spin.setRange(-10.0, 40.0)
        self.snr_spin.setSingleStep(1.0)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setSuffix(" dB")
        self.snr_spin.setValue(self.snr_db)
        self.snr_spin.valueChanged.connect(lambda v: setattr(self, 'snr_db', v))
        awgn_ctrl_layout.addWidget(self.snr_spin)

        self.awgn_controls.setEnabled(self.awgn_enabled)
        layout.addWidget(self.awgn_controls)

        def _on_awgn_toggled(checked):
            self.awgn_enabled = checked
            self.awgn_controls.setEnabled(checked)
            self.snr_spin.vary_checkbox.setEnabled(checked)
            if not checked:
                self.snr_spin.vary_checkbox.setChecked(False)
        self.awgn_toggle.toggled.connect(_on_awgn_toggled)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # Amplitude & Phase Shift
        amp_phase_header = QHBoxLayout()
        amp_phase_left = QVBoxLayout()
        amp_phase_title = QLabel("Amplitude & Phase Shift")
        amp_phase_title.setProperty("class", "section-title")
        amp_phase_desc = QLabel("Scale amplitude and rotate phase")
        amp_phase_desc.setProperty("class", "section-subtitle")
        amp_phase_left.addWidget(amp_phase_title)
        amp_phase_left.addWidget(amp_phase_desc)
        amp_phase_header.addLayout(amp_phase_left)
        amp_phase_header.addStretch()
        self.amp_phase_toggle = ToggleSwitch()
        self.amp_phase_toggle.setChecked(self.amp_phase_enabled)
        amp_phase_header.addWidget(self.amp_phase_toggle)
        layout.addLayout(amp_phase_header)

        # Amp/Phase controls — RangeSpinBoxes inside the grey-out wrapper.
        self.amp_phase_controls = QWidget()
        ap_ctrl_layout = QVBoxLayout(self.amp_phase_controls)
        ap_ctrl_layout.setContentsMargins(0, 0, 0, 0)

        self.amplitude_spin = RangeSpinBox()
        self.amplitude_spin.setRange(0.0, 5.0)
        self.amplitude_spin.setSingleStep(0.1)
        self.amplitude_spin.setDecimals(2)
        self.amplitude_spin.setValue(self.amplitude)
        self.amplitude_spin.valueChanged.connect(lambda v: setattr(self, 'amplitude', v))
        amp_header = QHBoxLayout()
        amp_header.addWidget(QLabel("Amplitude Scaling"))
        amp_header.addStretch()
        amp_header.addWidget(self.amplitude_spin.vary_checkbox)
        ap_ctrl_layout.addLayout(amp_header)
        ap_ctrl_layout.addWidget(self.amplitude_spin)

        self.phase_spin = RangeSpinBox()
        self.phase_spin.setRange(-180.0, 180.0)
        self.phase_spin.setSingleStep(5.0)
        self.phase_spin.setDecimals(1)
        self.phase_spin.setSuffix("°")
        self.phase_spin.setValue(self.phase_deg)
        self.phase_spin.valueChanged.connect(lambda v: setattr(self, 'phase_deg', v))
        phase_header = QHBoxLayout()
        phase_header.addWidget(QLabel("Phase Shift (degrees)"))
        phase_header.addStretch()
        phase_header.addWidget(self.phase_spin.vary_checkbox)
        ap_ctrl_layout.addLayout(phase_header)
        ap_ctrl_layout.addWidget(self.phase_spin)

        self.amp_phase_controls.setEnabled(self.amp_phase_enabled)
        layout.addWidget(self.amp_phase_controls)

        def _on_amp_phase_toggled(checked):
            self.amp_phase_enabled = checked
            self.amp_phase_controls.setEnabled(checked)
        self.amp_phase_toggle.toggled.connect(_on_amp_phase_toggled)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        # Frequency Shift (CFO)
        freq_shift_header = QHBoxLayout()
        freq_shift_left = QVBoxLayout()
        freq_shift_title = QLabel("Frequency Offset (CFO)")
        freq_shift_title.setProperty("class", "section-title")
        freq_shift_desc = QLabel("Simulate carrier frequency offset from oscillator mismatch")
        freq_shift_desc.setProperty("class", "section-subtitle")
        freq_shift_left.addWidget(freq_shift_title)
        freq_shift_left.addWidget(freq_shift_desc)
        freq_shift_header.addLayout(freq_shift_left)
        freq_shift_header.addStretch()
        self.freq_shift_toggle = ToggleSwitch()
        self.freq_shift_toggle.setChecked(self.freq_shift_enabled)
        freq_shift_header.addWidget(self.freq_shift_toggle)
        layout.addLayout(freq_shift_header)

        # Freq-shift control: RangeSpinBox inside the grey-out wrapper.  Units are
        # Hz rather than the MHz this tab used before, matching what the bulk
        # metadata records so a swept offset is unambiguous.
        self.freq_shift_controls = QWidget()
        fs_ctrl_layout = QVBoxLayout(self.freq_shift_controls)
        fs_ctrl_layout.setContentsMargins(0, 0, 0, 0)

        self.freq_shift_spin = RangeSpinBox()
        freq_shift_header_row = QHBoxLayout()
        freq_shift_header_row.addWidget(QLabel("Frequency Offset (Hz)"))
        freq_shift_header_row.addStretch()
        freq_shift_header_row.addWidget(self.freq_shift_spin.vary_checkbox)
        fs_ctrl_layout.addLayout(freq_shift_header_row)
        self.freq_shift_spin.setRange(-100e6, 100e6)
        self.freq_shift_spin.setSingleStep(1000.0)
        self.freq_shift_spin.setDecimals(0)
        self.freq_shift_spin.setSuffix(" Hz")
        self.freq_shift_spin.setValue(self.freq_shift_hz)
        self.freq_shift_spin.valueChanged.connect(lambda v: setattr(self, 'freq_shift_hz', v))
        fs_ctrl_layout.addWidget(self.freq_shift_spin)

        self.freq_shift_controls.setEnabled(self.freq_shift_enabled)
        layout.addWidget(self.freq_shift_controls)

        def _on_freq_shift_toggled(checked):
            self.freq_shift_enabled = checked
            self.freq_shift_controls.setEnabled(checked)
            self.freq_shift_spin.vary_checkbox.setEnabled(checked)
            if not checked:
                self.freq_shift_spin.vary_checkbox.setChecked(False)
        self.freq_shift_toggle.toggled.connect(_on_freq_shift_toggled)

        # Sync initial enabled state for vary checkboxes
        self.snr_spin.vary_checkbox.setEnabled(self.awgn_enabled)
        self.amplitude_spin.vary_checkbox.setEnabled(self.amp_phase_enabled)
        self.phase_spin.vary_checkbox.setEnabled(self.amp_phase_enabled)
        self.freq_shift_spin.vary_checkbox.setEnabled(self.freq_shift_enabled)

        return page

    # ---- Stochastic TDL page ----
    def _create_stochastic_page(self):
        page, layout = self._make_scroll_page()

        # --- Header ---
        stoch_title = QLabel("TDL Fading Channel")
        stoch_title.setProperty("class", "section-title")
        stoch_desc = QLabel("Stochastic time-varying multipath channel (requires TensorFlow + Sionna)")
        stoch_desc.setProperty("class", "section-subtitle")
        layout.addWidget(stoch_title)
        layout.addWidget(stoch_desc)

        # --- Channel Parameters (grid) ---
        ch_grid = QGridLayout()
        ch_grid.setHorizontalSpacing(12)
        ch_grid.setVerticalSpacing(8)
        ch_grid.setColumnStretch(1, 1)

        ch_grid.addWidget(QLabel("TDL Profile"), 0, 0)
        self.tdl_profile_combo = QComboBox()
        self.tdl_profile_combo.addItems(["A", "B", "C", "D", "E"])
        self.tdl_profile_combo.setCurrentText(self.tdl_profile)
        self.tdl_profile_combo.currentTextChanged.connect(
            lambda v: setattr(self, 'tdl_profile', v))
        ch_grid.addWidget(self.tdl_profile_combo, 0, 1)

        self.delay_spread_spin = RangeSpinBox()
        self.delay_spread_spin.setRange(1.0, 1000.0)
        self.delay_spread_spin.setSingleStep(10.0)
        self.delay_spread_spin.setDecimals(1)
        self.delay_spread_spin.setSuffix(" ns")
        self.delay_spread_spin.setValue(self.delay_spread_ns)
        self.delay_spread_spin.valueChanged.connect(
            lambda v: setattr(self, 'delay_spread_ns', v))
        delay_spread_label_widget = QWidget()
        delay_spread_label_lay = QHBoxLayout(delay_spread_label_widget)
        delay_spread_label_lay.setContentsMargins(0, 0, 0, 0)
        delay_spread_label_lay.addWidget(QLabel("Delay Spread (ns)"))
        delay_spread_label_lay.addWidget(self.delay_spread_spin.vary_checkbox)
        ch_grid.addWidget(delay_spread_label_widget, 1, 0)
        ch_grid.addWidget(self.delay_spread_spin, 1, 1)

        layout.addLayout(ch_grid)

        # --- Separator ---
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # --- Noise & Reproducibility ---
        stoch_snr_header = QHBoxLayout()
        stoch_snr_header.addWidget(QLabel("SNR (dB)"))
        stoch_snr_header.addStretch()
        self.stoch_snr_spin = RangeSpinBox()
        stoch_snr_header.addWidget(self.stoch_snr_spin.vary_checkbox)
        layout.addLayout(stoch_snr_header)
        self.stoch_snr_spin.setRange(-10.0, 40.0)
        self.stoch_snr_spin.setSingleStep(1.0)
        self.stoch_snr_spin.setDecimals(1)
        self.stoch_snr_spin.setSuffix(" dB")
        self.stoch_snr_spin.setValue(self.stoch_snr_db)
        self.stoch_snr_spin.valueChanged.connect(lambda v: setattr(self, 'stoch_snr_db', v))
        layout.addWidget(self.stoch_snr_spin)


        # --- Separator ---
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        # --- Dataset Info (read-only grid) ---
        info_title = QLabel("Dataset Info")
        info_title.setProperty("class", "section-title")
        layout.addWidget(info_title)

        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(12)
        info_grid.setVerticalSpacing(6)
        info_grid.setColumnStretch(1, 1)

        fc_label = QLabel("Carrier Frequency (MHz)")
        fc_label.setProperty("class", "section-subtitle")
        info_grid.addWidget(fc_label, 0, 0)
        self.carrier_freq_display = QLabel("--")
        self.carrier_freq_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_grid.addWidget(self.carrier_freq_display, 0, 1)

        sr_label = QLabel("Sample Rate (MHz)")
        sr_label.setProperty("class", "section-subtitle")
        info_grid.addWidget(sr_label, 1, 0)
        self.sample_rate_display = QLabel("--")
        self.sample_rate_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_grid.addWidget(self.sample_rate_display, 1, 1)

        out_label = QLabel("Output Samples")
        out_label.setProperty("class", "section-subtitle")
        info_grid.addWidget(out_label, 2, 0)
        self.output_samples_spin = QSpinBox()
        self.output_samples_spin.setRange(128, 65536)
        self.output_samples_spin.setSingleStep(256)
        self.output_samples_spin.setValue(self.output_num_samples)
        self.output_samples_spin.valueChanged.connect(
            lambda v: setattr(self, 'output_num_samples', v))
        info_grid.addWidget(self.output_samples_spin, 2, 1)

        layout.addLayout(info_grid)

        # --- Multi-Channel Output ---
        sep_mc = QFrame()
        sep_mc.setFrameShape(QFrame.HLine)
        sep_mc.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep_mc)

        mc_title = QLabel("Multi-Channel Array Output")
        mc_title.setProperty("class", "section-title")
        layout.addWidget(mc_title)

        self.stoch_multi_channel_cb = QCheckBox("Enable Multi-Channel Output")
        self.stoch_multi_channel_cb.setToolTip(
            "Return all RX antenna outputs instead of selecting a single one"
        )
        self.stoch_multi_channel_cb.toggled.connect(self._on_stoch_multi_channel_toggled)
        layout.addWidget(self.stoch_multi_channel_cb)

        stoch_mc_grid = QGridLayout()
        stoch_mc_grid.setHorizontalSpacing(12)
        stoch_mc_grid.setVerticalSpacing(8)
        stoch_mc_grid.setColumnStretch(1, 1)

        stoch_mc_grid.addWidget(QLabel("Num TX Antennas"), 0, 0)
        self.stoch_num_tx_ant_spin = QSpinBox()
        self.stoch_num_tx_ant_spin.setRange(1, 64)
        self.stoch_num_tx_ant_spin.setValue(1)
        self.stoch_num_tx_ant_spin.setEnabled(False)
        stoch_mc_grid.addWidget(self.stoch_num_tx_ant_spin, 0, 1)

        stoch_mc_grid.addWidget(QLabel("Num RX Antennas"), 1, 0)
        self.stoch_num_rx_ant_spin = QSpinBox()
        self.stoch_num_rx_ant_spin.setRange(1, 64)
        self.stoch_num_rx_ant_spin.setValue(1)
        self.stoch_num_rx_ant_spin.setEnabled(False)
        stoch_mc_grid.addWidget(self.stoch_num_rx_ant_spin, 1, 1)

        layout.addLayout(stoch_mc_grid)

        save_mode_row = QHBoxLayout()
        save_mode_row.addWidget(QLabel("Save Mode"))
        self.stoch_save_mode_combo = QComboBox()
        self.stoch_save_mode_combo.addItems([
            "Single Antenna",
            "All Antennas (Separate Files)",
            "All Antennas (Stacked)",
        ])
        self.stoch_save_mode_combo.setEnabled(False)
        save_mode_row.addWidget(self.stoch_save_mode_combo)
        layout.addLayout(save_mode_row)

        self.stoch_output_shape_label = QLabel("")
        self.stoch_output_shape_label.setProperty("class", "section-subtitle")
        layout.addWidget(self.stoch_output_shape_label)

        return page

    # ---- Measured Channel page ----
    def _create_measured_page(self):
        """Import real recorded channel impulse responses and apply them."""
        page, layout = self._make_scroll_page()

        title = QLabel("Measured Channel")
        title.setProperty("class", "section-title")
        desc = QLabel("Augment using channel impulse responses recorded from "
                      "real hardware")
        desc.setProperty("class", "section-subtitle")
        desc.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(desc)

        # --- Import ---
        import_title = QLabel("Channel Bank")
        import_title.setProperty("class", "section-title")
        layout.addWidget(import_title)

        dom_row = QHBoxLayout()
        dom_row.addWidget(QLabel("Interpret as"))
        self.meas_domain_combo = QComboBox()
        self.meas_domain_combo.addItems([
            "Auto-detect", "Impulse response h[n]", "Frequency response H(f)"])
        self.meas_domain_combo.setToolTip(
            "OFDM channel estimators output a value per subcarrier — a frequency "
            "response — which is converted to an impulse response on import. "
            "Auto-detect compares how concentrated the data is before and after "
            "an inverse FFT; override here if it guesses wrong.")
        dom_row.addWidget(self.meas_domain_combo, 1)
        layout.addLayout(dom_row)

        # No .bin format control here on purpose: .npy/.mat/.sigmf describe
        # their own sample type, so only raw .bin needs one.  It is asked for
        # on demand in _measured_bin_dtype(), when a .bin is actually being
        # imported, rather than sitting on screen for everyone else.

        btn_row = QHBoxLayout()
        self.meas_import_files_btn = QPushButton("Import Files…")
        self.meas_import_files_btn.clicked.connect(self._measured_import_files)
        btn_row.addWidget(self.meas_import_files_btn)

        self.meas_import_folder_btn = QPushButton("Import Folder…")
        self.meas_import_folder_btn.clicked.connect(self._measured_import_folder)
        btn_row.addWidget(self.meas_import_folder_btn)
        layout.addLayout(btn_row)

        self.meas_clear_btn = QPushButton("Clear Bank")
        self.meas_clear_btn.clicked.connect(self._measured_clear_bank)
        layout.addWidget(self.meas_clear_btn)

        layout.addWidget(QLabel("Loaded channels"))
        self.meas_channel_list = QListWidget()
        self.meas_channel_list.setMinimumHeight(140)
        self.meas_channel_list.currentRowChanged.connect(
            self._measured_on_selection_changed)
        layout.addWidget(self.meas_channel_list)

        self.meas_detail_label = QLabel("No channels loaded")
        self.meas_detail_label.setProperty("class", "section-subtitle")
        self.meas_detail_label.setWordWrap(True)
        layout.addWidget(self.meas_detail_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # --- Application options ---
        opt_title = QLabel("Options")
        opt_title.setProperty("class", "section-title")
        layout.addWidget(opt_title)

        self.meas_normalize_cb = QCheckBox("Normalize channel to unit energy")
        self.meas_normalize_cb.setChecked(True)
        self.meas_normalize_cb.setToolTip(
            "Keeps the signal's overall power unchanged so the channel alters "
            "its shape rather than its level. Turn off to keep the recorded "
            "path loss.")
        layout.addWidget(self.meas_normalize_cb)

        self.meas_awgn_cb = QCheckBox("Add AWGN after the channel")
        self.meas_awgn_cb.setChecked(False)
        layout.addWidget(self.meas_awgn_cb)

        # The SNR row lives in its own container so the checkbox disables the
        # label along with the spin box, matching how the AWGN page gates its
        # sections. Disabling the spin box alone left "SNR (dB)" at full
        # contrast, which read as an editable setting.
        self.meas_snr_controls = QWidget()
        snr_row = QHBoxLayout(self.meas_snr_controls)
        snr_row.setContentsMargins(0, 0, 0, 0)
        snr_row.addWidget(QLabel("SNR (dB)"))
        self.meas_snr_spin = QDoubleSpinBox()
        self.meas_snr_spin.setRange(-10.0, 40.0)
        self.meas_snr_spin.setValue(20.0)
        self.meas_snr_spin.setDecimals(1)
        snr_row.addWidget(self.meas_snr_spin, 1)
        self.meas_snr_controls.setEnabled(False)
        self.meas_awgn_cb.toggled.connect(self.meas_snr_controls.setEnabled)
        layout.addWidget(self.meas_snr_controls)

        self.meas_random_cb = QCheckBox("Pick a random channel from the bank")
        self.meas_random_cb.setToolTip(
            "Use a randomly chosen channel on each Apply, instead of the "
            "selected one — handy for building a varied training set.")
        layout.addWidget(self.meas_random_cb)

        return page

    # ---- Measured Channel handlers ----

    def _measured_bin_dtype(self, paths):
        """Sample layout for raw .bin, asked for only when one is present.

        .npy/.mat/.sigmf carry their own type, so most imports never need this.
        A raw .bin has no header, and a wrong guess yields plausible-looking
        garbage rather than an error, so it has to be asked rather than assumed.
        Returns None if the user cancels.
        """
        if not any(str(p).lower().endswith(".bin") for p in paths):
            return BIN_DTYPES[next(iter(BIN_DTYPES))]      # unused; keeps the call simple

        options = list(BIN_DTYPES.keys())
        choice, accepted = QInputDialog.getItem(
            self, "Raw .bin sample format",
            "These files have no header, so the sample layout must be given:",
            options, 0, False)
        if not accepted:
            return None
        return BIN_DTYPES[choice]

    def _measured_domain(self):
        return DOMAINS[self.meas_domain_combo.currentIndex()]

    def _measured_import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Channel Files", "",
            "Channel data (*.npy *.mat *.sigmf-meta *.sigmf *.bin);;All Files (*)")
        if not paths:
            return
        bin_dtype = self._measured_bin_dtype(paths)
        if bin_dtype is None:
            return                                          # user cancelled
        added, errors = 0, {}
        for p in paths:
            try:
                added += self.channel_bank.add_file(
                    p, bin_dtype=bin_dtype,
                    domain=self._measured_domain())
            except Exception as exc:                       # noqa: BLE001
                errors[os.path.basename(p)] = str(exc)
        self._measured_report_import(added, errors)

    def _measured_import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Import Channel Folder")
        if not folder:
            return
        # Only prompt if the folder actually holds .bin files
        try:
            contents = [str(p) for p in Path(folder).iterdir()]
        except OSError:
            contents = []
        bin_dtype = self._measured_bin_dtype(contents)
        if bin_dtype is None:
            return                                          # user cancelled
        try:
            added, errors = self.channel_bank.add_folder(
                folder, bin_dtype=bin_dtype,
                domain=self._measured_domain())
        except Exception as exc:                           # noqa: BLE001
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        self._measured_report_import(added, errors)

    def _measured_report_import(self, added, errors):
        """Refresh the list and tell the user what did and did not load."""
        self._measured_refresh_list()
        if added and not errors:
            QMessageBox.information(
                self, "Channels imported",
                f"Added {added} channel(s). The bank now holds "
                f"{len(self.channel_bank)}.")
        elif added and errors:
            detail = "\n".join(f"\n• {k}\n    {v}" for k, v in errors.items())
            QMessageBox.warning(
                self, "Imported with problems",
                f"Added {added} channel(s); {len(errors)} file(s) could not be "
                f"read:\n{detail}")
        elif errors:
            detail = "\n".join(f"\n• {k}\n    {v}" for k, v in errors.items())
            QMessageBox.critical(self, "Nothing imported",
                                 f"No channels could be read:\n{detail}")
        else:
            QMessageBox.information(self, "Nothing imported",
                                    "No supported channel files were found.")

    def _measured_clear_bank(self):
        self.channel_bank.clear()
        self._measured_refresh_list()

    def _measured_refresh_list(self):
        self.meas_channel_list.clear()
        for ch in self.channel_bank:
            self.meas_channel_list.addItem(ch.name)
        if len(self.channel_bank):
            self.meas_channel_list.setCurrentRow(0)
        else:
            self.meas_detail_label.setText("No channels loaded")

    def _measured_on_selection_changed(self, row):
        ch = self._measured_selected_channel(row)
        if ch is None:
            self.meas_detail_label.setText("No channels loaded")
            return
        src = os.path.basename(ch.source_path) if ch.source_path else "—"
        dom = ch.metadata.get("domain", "time")
        dom_txt = " (converted from H(f))" if dom == "frequency" else ""
        self.meas_detail_label.setText(f"{ch.describe()}{dom_txt}\nfrom {src}")

    def _measured_selected_channel(self, row=None):
        if not len(self.channel_bank):
            return None
        if row is None:
            row = self.meas_channel_list.currentRow()
        channels = self.channel_bank.channels
        if row is None or row < 0 or row >= len(channels):
            return None
        return channels[row]

    def _build_measured_block(self):
        """Construct the augmentation block, or None with a reason shown."""
        if not len(self.channel_bank):
            QMessageBox.warning(
                self, "No channels", "Import channel files before applying a "
                "measured channel.")
            return None

        if self.meas_random_cb.isChecked():
            channels = self.channel_bank.channels
            channel = channels[np.random.default_rng(experiment_seed()).integers(len(channels))]
        else:
            channel = self._measured_selected_channel()
            if channel is None:
                QMessageBox.warning(self, "No channel selected",
                                    "Select a channel from the bank.")
                return None

        return MeasuredChannelAugmentation(
            channel,
            normalize=self.meas_normalize_cb.isChecked(),
            snr_db=(self.meas_snr_spin.value()
                    if self.meas_awgn_cb.isChecked() else None),
            seed=experiment_seed(),
        )

    # ---- Ray Tracing page (full controls) ----
    def _create_rt_page(self):
        """Create the RT page with all controls, styled to match AWGN/Stochastic pages."""
        page, layout = self._make_scroll_page()

        # --- Header ---
        rt_title = QLabel("Sionna Ray Tracing Channel")
        rt_title.setProperty("class", "section-title")
        rt_desc = QLabel("Deterministic channel from 3D scene ray tracing")
        rt_desc.setProperty("class", "section-subtitle")
        layout.addWidget(rt_title)
        layout.addWidget(rt_desc)

        # --- Scene ---
        self.rt_scene_label = QLabel("No scene loaded")
        self.rt_scene_label.setWordWrap(True)
        self.rt_scene_label.setProperty("class", "section-subtitle")
        layout.addWidget(self.rt_scene_label)

        self.rt_preset_combo = QComboBox()
        self.rt_preset_combo.addItem("Bundled scene...", None)
        for scene in available_scenes():
            self.rt_preset_combo.addItem(scene["name"], scene)
        self.rt_preset_combo.currentIndexChanged.connect(self._rt_load_preset)
        layout.addWidget(self.rt_preset_combo)

        self.rt_load_scene_btn = QPushButton("Load Scene...")
        self.rt_load_scene_btn.clicked.connect(self._rt_load_scene)
        layout.addWidget(self.rt_load_scene_btn)

        # --- Separator ---
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep1)

        # --- Simulation Parameters (grid) ---
        param_title = QLabel("Simulation Parameters")
        param_title.setProperty("class", "section-title")
        layout.addWidget(param_title)

        param_grid = QGridLayout()
        param_grid.setHorizontalSpacing(12)
        param_grid.setVerticalSpacing(8)
        param_grid.setColumnStretch(1, 1)

        fc_label = QLabel("Center Frequency (MHz)")
        fc_label.setProperty("class", "section-subtitle")
        param_grid.addWidget(fc_label, 0, 0)
        self.rt_freq_display = QLabel("--")
        self.rt_freq_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        param_grid.addWidget(self.rt_freq_display, 0, 1)

        sr_label = QLabel("Sample Rate (MHz)")
        sr_label.setProperty("class", "section-subtitle")
        param_grid.addWidget(sr_label, 1, 0)
        self.rt_sample_rate_display = QLabel("--")
        self.rt_sample_rate_display.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        param_grid.addWidget(self.rt_sample_rate_display, 1, 1)

        param_grid.addWidget(QLabel("Max Depth"), 2, 0)
        self.rt_depth_spin = QSpinBox()
        self.rt_depth_spin.setRange(1, 20)
        self.rt_depth_spin.setValue(3)
        self.rt_depth_spin.valueChanged.connect(self._rt_push_max_depth)
        param_grid.addWidget(self.rt_depth_spin, 2, 1)

        param_grid.addWidget(QLabel("Num Samples"), 3, 0)
        self.rt_samples_spin = QSpinBox()
        self.rt_samples_spin.setRange(1000, 10_000_000)
        self.rt_samples_spin.setSingleStep(100_000)
        self.rt_samples_spin.setValue(1_000_000)
        self.rt_samples_spin.valueChanged.connect(self._rt_push_num_samples)
        param_grid.addWidget(self.rt_samples_spin, 3, 1)

        layout.addLayout(param_grid)

        # --- Separator ---
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep2)

        # --- Transceivers ---
        trx_title = QLabel("Transceivers")
        trx_title.setProperty("class", "section-title")
        layout.addWidget(trx_title)

        # TX Position
        layout.addWidget(QLabel("TX Position"))
        self.rt_tx_x = QDoubleSpinBox()
        self.rt_tx_y = QDoubleSpinBox()
        self.rt_tx_z = QDoubleSpinBox()
        for spin, default in [(self.rt_tx_x, 0.0), (self.rt_tx_y, 0.0), (self.rt_tx_z, 10.0)]:
            spin.setRange(-10000, 10000)
            spin.setDecimals(1)
            spin.setValue(default)
            spin.valueChanged.connect(self._rt_push_tx_position)
        tx_pos_row = QHBoxLayout()
        tx_pos_row.setSpacing(4)
        tx_pos_row.addWidget(QLabel("X:"))
        tx_pos_row.addWidget(self.rt_tx_x)
        tx_pos_row.addWidget(QLabel("Y:"))
        tx_pos_row.addWidget(self.rt_tx_y)
        tx_pos_row.addWidget(QLabel("Z:"))
        tx_pos_row.addWidget(self.rt_tx_z)
        layout.addLayout(tx_pos_row)

        self.rt_place_tx_btn = QPushButton("Place TX")
        self.rt_place_tx_btn.setCheckable(True)
        self.rt_place_tx_btn.toggled.connect(self._rt_on_place_tx)
        layout.addWidget(self.rt_place_tx_btn)

        # RX Position
        layout.addWidget(QLabel("RX Position"))
        self.rt_rx_x = QDoubleSpinBox()
        self.rt_rx_y = QDoubleSpinBox()
        self.rt_rx_z = QDoubleSpinBox()
        for spin, default in [(self.rt_rx_x, 50.0), (self.rt_rx_y, 0.0), (self.rt_rx_z, 1.5)]:
            spin.setRange(-10000, 10000)
            spin.setDecimals(1)
            spin.setValue(default)
            spin.valueChanged.connect(self._rt_push_rx_position)
        rx_pos_row = QHBoxLayout()
        rx_pos_row.setSpacing(4)
        rx_pos_row.addWidget(QLabel("X:"))
        rx_pos_row.addWidget(self.rt_rx_x)
        rx_pos_row.addWidget(QLabel("Y:"))
        rx_pos_row.addWidget(self.rt_rx_y)
        rx_pos_row.addWidget(QLabel("Z:"))
        rx_pos_row.addWidget(self.rt_rx_z)
        layout.addLayout(rx_pos_row)

        self.rt_place_rx_btn = QPushButton("Place RX")
        self.rt_place_rx_btn.setCheckable(True)
        self.rt_place_rx_btn.toggled.connect(self._rt_on_place_rx)
        layout.addWidget(self.rt_place_rx_btn)

        # --- Separator ---
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.HLine)
        sep3.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep3)

        # --- Antenna Arrays (grid) ---
        ant_title = QLabel("Antenna Arrays")
        ant_title.setProperty("class", "section-title")
        layout.addWidget(ant_title)

        ant_grid = QGridLayout()
        ant_grid.setHorizontalSpacing(12)
        ant_grid.setVerticalSpacing(8)
        ant_grid.setColumnStretch(1, 1)

        ant_grid.addWidget(QLabel("TX Rows"), 0, 0)
        self.rt_tx_ant_rows = QSpinBox()
        self.rt_tx_ant_rows.setRange(1, 16)
        self.rt_tx_ant_rows.setValue(6)
        self.rt_tx_ant_rows.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_tx_ant_rows, 0, 1)

        ant_grid.addWidget(QLabel("TX Cols"), 1, 0)
        self.rt_tx_ant_cols = QSpinBox()
        self.rt_tx_ant_cols.setRange(1, 16)
        self.rt_tx_ant_cols.setValue(6)
        self.rt_tx_ant_cols.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_tx_ant_cols, 1, 1)

        ant_grid.addWidget(QLabel("TX Pattern"), 2, 0)
        self.rt_tx_ant_pattern = QComboBox()
        self.rt_tx_ant_pattern.addItems(["tr38901", "iso"])
        self.rt_tx_ant_pattern.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_tx_ant_pattern, 2, 1)

        ant_grid.addWidget(QLabel("TX Polarization"), 3, 0)
        self.rt_tx_ant_pol = QComboBox()
        self.rt_tx_ant_pol.addItems(["V", "H", "cross"])
        self.rt_tx_ant_pol.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_tx_ant_pol, 3, 1)

        ant_grid.addWidget(QLabel("RX Rows"), 4, 0)
        self.rt_rx_ant_rows = QSpinBox()
        self.rt_rx_ant_rows.setRange(1, 16)
        self.rt_rx_ant_rows.setValue(6)
        self.rt_rx_ant_rows.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_rx_ant_rows, 4, 1)

        ant_grid.addWidget(QLabel("RX Cols"), 5, 0)
        self.rt_rx_ant_cols = QSpinBox()
        self.rt_rx_ant_cols.setRange(1, 16)
        self.rt_rx_ant_cols.setValue(6)
        self.rt_rx_ant_cols.valueChanged.connect(self._rt_push_antennas)
        ant_grid.addWidget(self.rt_rx_ant_cols, 5, 1)

        ant_grid.addWidget(QLabel("RX Pattern"), 6, 0)
        self.rt_rx_ant_pattern = QComboBox()
        self.rt_rx_ant_pattern.addItems(["iso", "tr38901"])
        self.rt_rx_ant_pattern.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_rx_ant_pattern, 6, 1)

        ant_grid.addWidget(QLabel("RX Polarization"), 7, 0)
        self.rt_rx_ant_pol = QComboBox()
        self.rt_rx_ant_pol.addItems(["V", "H", "cross"])
        self.rt_rx_ant_pol.currentTextChanged.connect(lambda _: self._rt_push_antennas())
        ant_grid.addWidget(self.rt_rx_ant_pol, 7, 1)

        layout.addLayout(ant_grid)

        # --- Separator ---
        sep4 = QFrame()
        sep4.setFrameShape(QFrame.HLine)
        sep4.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep4)

        # --- Path Solver ---
        solver_title = QLabel("Path Solver")
        solver_title.setProperty("class", "section-title")
        layout.addWidget(solver_title)

        solver_toggles = [
            ("Line of Sight",       "rt_cb_los"),
            ("Specular Reflection", "rt_cb_specular"),
            ("Diffuse Reflection",  "rt_cb_diffuse"),
            ("Refraction",          "rt_cb_refraction"),
            ("Synthetic Array",     "rt_cb_synthetic"),
        ]
        for label_text, attr_name in solver_toggles:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addStretch()
            toggle = ToggleSwitch()
            toggle.setChecked(True)
            toggle.toggled.connect(lambda _: self._rt_push_solver())
            setattr(self, attr_name, toggle)
            row.addWidget(toggle)
            layout.addLayout(row)

        # --- Separator ---
        sep5 = QFrame()
        sep5.setFrameShape(QFrame.HLine)
        sep5.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep5)

        # --- Channel Parameters (grid) ---
        ch_title = QLabel("Channel Parameters")
        ch_title.setProperty("class", "section-title")
        layout.addWidget(ch_title)

        ch_grid = QGridLayout()
        ch_grid.setHorizontalSpacing(12)
        ch_grid.setVerticalSpacing(8)
        ch_grid.setColumnStretch(1, 1)

        ch_grid.addWidget(QLabel("Bandwidth (MHz)"), 0, 0)
        self.rt_bandwidth_spin = QDoubleSpinBox()
        self.rt_bandwidth_spin.setRange(0.01, 1000.0)
        self.rt_bandwidth_spin.setValue(5.0)
        self.rt_bandwidth_spin.setDecimals(2)
        ch_grid.addWidget(self.rt_bandwidth_spin, 0, 1)

        ch_grid.addWidget(QLabel("l_min"), 1, 0)
        self.rt_l_min_spin = QSpinBox()
        self.rt_l_min_spin.setRange(0, 10000)
        self.rt_l_min_spin.setValue(0)
        ch_grid.addWidget(self.rt_l_min_spin, 1, 1)

        ch_grid.addWidget(QLabel("l_max"), 2, 0)
        self.rt_l_max_spin = QSpinBox()
        self.rt_l_max_spin.setRange(1, 10000)
        self.rt_l_max_spin.setValue(200)
        ch_grid.addWidget(self.rt_l_max_spin, 2, 1)

        ch_grid.addWidget(QLabel("TX Power (dBm)"), 3, 0)
        self.rt_tx_power_spin = QDoubleSpinBox()
        self.rt_tx_power_spin.setRange(-30.0, 60.0)
        self.rt_tx_power_spin.setValue(30.0)
        self.rt_tx_power_spin.setDecimals(1)
        ch_grid.addWidget(self.rt_tx_power_spin, 3, 1)

        ch_grid.addWidget(QLabel("Noise Power (dBm)"), 4, 0)
        self.rt_noise_power_spin = QDoubleSpinBox()
        self.rt_noise_power_spin.setRange(-200.0, 0.0)
        self.rt_noise_power_spin.setValue(-108.0)
        self.rt_noise_power_spin.setDecimals(1)
        ch_grid.addWidget(self.rt_noise_power_spin, 4, 1)

        ch_grid.addWidget(QLabel("Waveform Length"), 5, 0)
        self.rt_waveform_length_spin = QSpinBox()
        self.rt_waveform_length_spin.setRange(64, 65536)
        self.rt_waveform_length_spin.setValue(1024)
        self.rt_waveform_length_spin.setSingleStep(256)
        ch_grid.addWidget(self.rt_waveform_length_spin, 5, 1)

        ch_grid.addWidget(QLabel("Waveform Padding"), 6, 0)
        self.rt_padding_combo = QComboBox()
        self.rt_padding_combo.addItems(["Zero Padded", "Tile / Repeat"])
        ch_grid.addWidget(self.rt_padding_combo, 6, 1)

        ch_grid.addWidget(QLabel("RX Antenna Index"), 7, 0)
        self.rt_rx_antenna_idx_spin = QSpinBox()
        self.rt_rx_antenna_idx_spin.setRange(0, 0)
        self.rt_rx_antenna_idx_spin.setValue(0)
        ch_grid.addWidget(self.rt_rx_antenna_idx_spin, 7, 1)

        layout.addLayout(ch_grid)

        # --- Multi-Channel Output ---
        sep_mc = QFrame()
        sep_mc.setFrameShape(QFrame.HLine)
        sep_mc.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep_mc)

        mc_title = QLabel("Multi-Channel Array Output")
        mc_title.setProperty("class", "section-title")
        layout.addWidget(mc_title)

        self.rt_multi_channel_cb = QCheckBox("Enable Multi-Channel Output")
        self.rt_multi_channel_cb.setToolTip(
            "Return all RX antenna outputs instead of selecting a single one"
        )
        self.rt_multi_channel_cb.toggled.connect(self._on_rt_multi_channel_toggled)
        layout.addWidget(self.rt_multi_channel_cb)

        save_mode_row = QHBoxLayout()
        save_mode_row.addWidget(QLabel("Save Mode"))
        self.rt_save_mode_combo = QComboBox()
        self.rt_save_mode_combo.addItems([
            "Single Antenna",
            "All Antennas (Separate Files)",
            "All Antennas (Stacked)",
        ])
        self.rt_save_mode_combo.setEnabled(False)
        save_mode_row.addWidget(self.rt_save_mode_combo)
        layout.addLayout(save_mode_row)

        self.rt_output_shape_label = QLabel("")
        self.rt_output_shape_label.setProperty("class", "section-subtitle")
        layout.addWidget(self.rt_output_shape_label)

        return page

    # ---- RT control push methods ----

    def _rt_load_scene(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Sionna Scene", "",
            "XML Files (*.xml);;All Files (*)",
        )
        if not path:
            return
        # Browsing to a file clears any bundled-scene selection
        self.rt_preset_combo.blockSignals(True)
        self.rt_preset_combo.setCurrentIndex(0)
        self.rt_preset_combo.blockSignals(False)
        self._rt_load_scene_path(path)

    def _rt_load_scene_path(self, path: str) -> bool:
        """Load *path* into the widget, resetting stale RT state.

        Shared by the file dialog and the bundled-scene picker.
        """
        if not self._ensure_sionna_widget():
            return False

        # Clear stale taps from a previous scene
        self.rt_last_taps = None
        self.rt_tap_grid = None

        # Re-enable compute button in case it was stuck from a prior error
        self.rt_compute_btn.setEnabled(True)
        self.rt_compute_btn.setText("Compute Paths")

        try:
            self.sionna_widget.load_scene(path)
            self.rt_scene_label.setText(os.path.basename(path))
            return True
        except Exception as e:
            self.rt_status_label.setText(f"Error loading scene: {e}")
            print(f"Failed to load scene: {e}")
            return False

    def _rt_load_preset(self, _index=None):
        """Load a bundled scene and apply its TX/RX and frequency defaults."""
        scene = self.rt_preset_combo.currentData()
        if scene is None:
            return
        if not self._rt_load_scene_path(str(scene["path"])):
            return

        for spin, value in zip(
            (self.rt_tx_x, self.rt_tx_y, self.rt_tx_z), scene["tx_position"]
        ):
            spin.setValue(value)
        for spin, value in zip(
            (self.rt_rx_x, self.rt_rx_y, self.rt_rx_z), scene["rx_position"]
        ):
            spin.setValue(value)

        # Push explicitly: setValue is a no-op (and emits nothing) when the
        # spinbox already holds the value, e.g. reloading the same preset.
        self._rt_push_tx_position()
        self._rt_push_rx_position()

        # Fallback carrier — _rt_compute_paths overrides this from the active
        # dataset's fc when one is selected.
        freq_ghz = scene.get("frequency_ghz")
        if freq_ghz is not None:
            try:
                self.sionna_widget.set_frequency_ghz(float(freq_ghz))
                if self._active_entry is None:
                    self.rt_freq_display.setText(f"{float(freq_ghz) * 1e3:.4g}")
            except Exception as e:
                print(f"Could not set preset frequency: {e}")

        self.rt_status_label.setText(scene.get("description", scene["name"]))

    def _rt_push_max_depth(self, val):
        if self.sionna_widget is not None:
            self.sionna_widget.set_max_depth(val)

    def _rt_push_num_samples(self, val):
        if self.sionna_widget is not None:
            self.sionna_widget.set_num_samples(val)

    def _rt_push_tx_position(self, _=None):
        if self.sionna_widget is not None:
            pos = [self.rt_tx_x.value(), self.rt_tx_y.value(), self.rt_tx_z.value()]
            self.sionna_widget.set_tx_position(pos)

    def _rt_push_rx_position(self, _=None):
        if self.sionna_widget is not None:
            pos = [self.rt_rx_x.value(), self.rt_rx_y.value(), self.rt_rx_z.value()]
            self.sionna_widget.set_rx_position(pos)

    def _rt_on_place_tx(self, on):
        if on:
            self.rt_place_rx_btn.blockSignals(True)
            self.rt_place_rx_btn.setChecked(False)
            self.rt_place_rx_btn.blockSignals(False)
            if self.sionna_widget is not None:
                self.sionna_widget.enter_placement_mode("tx")

    def _rt_on_place_rx(self, on):
        if on:
            self.rt_place_tx_btn.blockSignals(True)
            self.rt_place_tx_btn.setChecked(False)
            self.rt_place_tx_btn.blockSignals(False)
            if self.sionna_widget is not None:
                self.sionna_widget.enter_placement_mode("rx")

    def _rt_uncheck_placement_buttons(self):
        self.rt_place_tx_btn.blockSignals(True)
        self.rt_place_rx_btn.blockSignals(True)
        self.rt_place_tx_btn.setChecked(False)
        self.rt_place_rx_btn.setChecked(False)
        self.rt_place_tx_btn.blockSignals(False)
        self.rt_place_rx_btn.blockSignals(False)

    def _on_rt_multi_channel_toggled(self, checked):
        self.rt_rx_antenna_idx_spin.setEnabled(not checked)
        self.rt_save_mode_combo.setEnabled(checked)
        if not checked:
            self.rt_save_mode_combo.setCurrentIndex(0)
            self.rt_output_shape_label.setText("")

    def _on_stoch_multi_channel_toggled(self, checked):
        self.stoch_num_tx_ant_spin.setEnabled(checked)
        self.stoch_num_rx_ant_spin.setEnabled(checked)
        self.stoch_save_mode_combo.setEnabled(checked)
        if not checked:
            self.stoch_save_mode_combo.setCurrentIndex(0)
            self.stoch_output_shape_label.setText("")

    def _rt_push_antennas(self, _=None):
        if self.sionna_widget is None:
            return
        tx_cfg = {
            "num_rows": self.rt_tx_ant_rows.value(),
            "num_cols": self.rt_tx_ant_cols.value(),
            "vert_spacing": 0.5,
            "hori_spacing": 0.5,
            "pattern": self.rt_tx_ant_pattern.currentText(),
            "polarization": self.rt_tx_ant_pol.currentText(),
            "polarization_model": "tr38901_2",
        }
        rx_cfg = {
            "num_rows": self.rt_rx_ant_rows.value(),
            "num_cols": self.rt_rx_ant_cols.value(),
            "vert_spacing": 0.5,
            "hori_spacing": 0.5,
            "pattern": self.rt_rx_ant_pattern.currentText(),
            "polarization": self.rt_rx_ant_pol.currentText(),
            "polarization_model": "tr38901_2",
        }
        self.sionna_widget.set_antenna_arrays(tx_cfg, rx_cfg)
        # Update RX antenna index max
        max_idx = rx_cfg["num_rows"] * rx_cfg["num_cols"] - 1
        self.rt_rx_antenna_idx_spin.setMaximum(max(max_idx, 0))
        if self.rt_rx_antenna_idx_spin.value() > max_idx:
            self.rt_rx_antenna_idx_spin.setValue(0)

    def _rt_push_solver(self):
        if self.sionna_widget is None:
            return
        opts = {
            "line_of_sight": self.rt_cb_los.isChecked(),
            "specular_reflection": self.rt_cb_specular.isChecked(),
            "diffuse_reflection": self.rt_cb_diffuse.isChecked(),
            "refraction": self.rt_cb_refraction.isChecked(),
            "synthetic_array": self.rt_cb_synthetic.isChecked(),
        }
        self.sionna_widget.set_solver_options(opts)

    def _rt_get_channel_params_dict(self):
        params = {
            "bandwidth_hz": self.rt_bandwidth_spin.value() * 1e6,
            "l_min": self.rt_l_min_spin.value(),
            "l_max": self.rt_l_max_spin.value(),
            "tx_power_dbm": self.rt_tx_power_spin.value(),
            "noise_power_dbm": self.rt_noise_power_spin.value(),
            "waveform_length": self.rt_waveform_length_spin.value(),
            "zero_padding": self.rt_padding_combo.currentText(),
            "rx_antenna_index": self.rt_rx_antenna_idx_spin.value(),
        }
        if self._active_entry is not None:
            params["sample_rate_hz"] = float(self._active_entry["fs"])
        return params

    def _rt_push_channel_params(self):
        if self.sionna_widget is not None:
            self.sionna_widget.set_channel_params(self._rt_get_channel_params_dict())

    def _rt_compute_paths(self):
        if not self._ensure_sionna_widget():
            return

        self.rt_last_taps = None
        self.rt_tap_grid = None

        # Set engine frequency from dataset metadata before computing paths.
        entry = self._active_entry
        if entry:
            fc = entry.get('fc', None)
            if fc is not None:
                try:
                    self.sionna_widget.set_frequency_ghz(float(fc) / 1e9)
                except Exception as e:
                    self.rt_status_label.setText(
                        f"Warning: could not set frequency {float(fc)/1e6:.4g} MHz — {e}")

        # Push all current params before computing
        self._rt_push_channel_params()
        self.rt_compute_btn.setEnabled(False)
        self.rt_compute_btn.setText("Computing...")
        self.sionna_widget.compute_paths()

    # ---- RT viewport placement feedback ----

    def _on_rt_tx_placed(self, pos):
        """Viewport placed TX → update spinboxes."""
        self.rt_tx_x.blockSignals(True)
        self.rt_tx_y.blockSignals(True)
        self.rt_tx_z.blockSignals(True)
        self.rt_tx_x.setValue(pos[0])
        self.rt_tx_y.setValue(pos[1])
        self.rt_tx_z.setValue(pos[2])
        self.rt_tx_x.blockSignals(False)
        self.rt_tx_y.blockSignals(False)
        self.rt_tx_z.blockSignals(False)
        self._rt_uncheck_placement_buttons()

    def _on_rt_rx_placed(self, pos):
        """Viewport placed RX → update spinboxes."""
        self.rt_rx_x.blockSignals(True)
        self.rt_rx_y.blockSignals(True)
        self.rt_rx_z.blockSignals(True)
        self.rt_rx_x.setValue(pos[0])
        self.rt_rx_y.setValue(pos[1])
        self.rt_rx_z.setValue(pos[2])
        self.rt_rx_x.blockSignals(False)
        self.rt_rx_y.blockSignals(False)
        self.rt_rx_z.blockSignals(False)
        self._rt_uncheck_placement_buttons()

    # ---- subtab switching ----

    def _on_subtab_changed(self, idx):
        self.active_subtab = idx
        self.subtab_stack.setCurrentIndex(idx)

        is_rt = idx == 2
        self.rt_compute_btn.setVisible(is_rt)
        self.rt_status_label.setVisible(is_rt)

        if is_rt:
            # RT active: show viewport + comparison subtabs, hide plain comparison
            self.rt_right_container.setVisible(True)
            self.comparison_card.setVisible(False)
            self._ensure_sionna_widget()
        else:
            # AWGN / Stochastic: show plain comparison, hide RT container
            self.rt_right_container.setVisible(False)
            self.comparison_card.setVisible(True)

    # ---- Visualization (right) panel ----

    def create_visualization_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── RT right container (hidden by default) ───────────
        self.rt_right_container = QWidget()
        rt_container_layout = QVBoxLayout(self.rt_right_container)
        rt_container_layout.setContentsMargins(0, 0, 0, 0)
        rt_container_layout.setSpacing(8)

        # Pill toggle: 3D Viewport | Signal Comparison
        rt_pill_layout = QHBoxLayout()
        rt_pill_layout.setSpacing(8)
        self.rt_view_group = QButtonGroup(self)
        self.rt_view_group.setExclusive(True)

        self.rt_viewport_pill = QPushButton("3D Viewport")
        self.rt_viewport_pill.setObjectName("subtabPill")
        self.rt_viewport_pill.setCheckable(True)
        self.rt_viewport_pill.setChecked(True)

        self.rt_comparison_pill = QPushButton("Signal Comparison")
        self.rt_comparison_pill.setObjectName("subtabPill")
        self.rt_comparison_pill.setCheckable(True)

        self.rt_view_group.addButton(self.rt_viewport_pill, 0)
        self.rt_view_group.addButton(self.rt_comparison_pill, 1)

        rt_pill_layout.addWidget(self.rt_viewport_pill)
        rt_pill_layout.addWidget(self.rt_comparison_pill)
        rt_pill_layout.addStretch()
        rt_container_layout.addLayout(rt_pill_layout)

        # Stacked widget for viewport vs comparison
        self.rt_view_stack = QStackedWidget()

        # Index 0: viewport page (placeholder, filled by _ensure_sionna_widget)
        self.rt_viewport_page = QWidget()
        self.rt_viewport_page_layout = QVBoxLayout(self.rt_viewport_page)
        self.rt_viewport_page_layout.setContentsMargins(0, 0, 0, 0)
        self.rt_view_stack.addWidget(self.rt_viewport_page)

        # Index 1: RT comparison page
        rt_comparison_frame = QFrame()
        rt_comparison_frame.setObjectName("card")
        rt_comp_layout = QVBoxLayout(rt_comparison_frame)
        rt_comp_layout.setContentsMargins(24, 24, 24, 24)
        rt_comp_title = QLabel("Clean vs Augmented Signal")
        rt_comp_title.setProperty("class", "card-title")
        rt_comp_layout.addWidget(rt_comp_title)

        from mixedsignal_gui.widgets.comparison_widget import ComparisonWidget
        self.rt_comparison_plot = ComparisonWidget()
        rt_comp_layout.addWidget(self.rt_comparison_plot)
        self.rt_view_stack.addWidget(rt_comparison_frame)

        rt_container_layout.addWidget(self.rt_view_stack)

        self.rt_view_group.idClicked.connect(self.rt_view_stack.setCurrentIndex)

        self.rt_right_container.setVisible(False)
        layout.addWidget(self.rt_right_container)

        # ── Comparison card (visible for AWGN / Stochastic) ──
        self.comparison_card = QFrame()
        self.comparison_card.setObjectName("card")
        comparison_layout = QVBoxLayout(self.comparison_card)
        comparison_layout.setContentsMargins(24, 24, 24, 24)

        comparison_title = QLabel("Clean vs Augmented Signal")
        comparison_title.setProperty("class", "card-title")
        comparison_layout.addWidget(comparison_title)

        from mixedsignal_gui.widgets.comparison_widget import ComparisonWidget
        self.comparison_plot = ComparisonWidget()
        comparison_layout.addWidget(self.comparison_plot)

        layout.addWidget(self.comparison_card)

        return panel

    def _ensure_sionna_widget(self) -> bool:
        """Lazy-create the SionnaWidget (viewport_only) on first RT tab activation."""
        if self.sionna_widget is not None:
            return True
        try:
            from mixedsignal_gui.sionna_widget import SionnaWidget
            self.sionna_widget = SionnaWidget(viewport_only=True)
            self.rt_viewport_page_layout.addWidget(self.sionna_widget)

            # Connect signals
            self.sionna_widget.paths_computed.connect(self._on_rt_paths_computed)
            self.sionna_widget.taps_computed.connect(self._on_rt_taps_computed)
            self.sionna_widget.error_occurred.connect(self._on_rt_error)
            self.sionna_widget.scene_loaded.connect(
                lambda path: self.rt_status_label.setText(f"Scene: {path.rsplit('/', 1)[-1]}"))

            # Viewport placement → update position spinboxes
            self.sionna_widget.tx_placed.connect(self._on_rt_tx_placed)
            self.sionna_widget.rx_placed.connect(self._on_rt_rx_placed)

            # Push current control values to the widget
            self._rt_push_max_depth(self.rt_depth_spin.value())
            self._rt_push_num_samples(self.rt_samples_spin.value())
            self._rt_push_tx_position()
            self._rt_push_rx_position()
            self._rt_push_antennas()
            self._rt_push_solver()
            self._rt_push_channel_params()

            return True
        except ImportError as e:
            QMessageBox.warning(
                self, "Missing Dependencies",
                f"Ray Tracing requires TensorFlow and Sionna RT.\n\n{e}"
            )
            return False

    def _on_rt_paths_computed(self, channel_params):
        if channel_params.num_paths == 0:
            self.rt_status_label.setText(
                "Paths: 0 found — check TX/RX positions are inside the scene")
        else:
            self.rt_status_label.setText(f"Paths: {channel_params.num_paths} found")
        self.rt_compute_btn.setEnabled(True)
        self.rt_compute_btn.setText("Compute Paths")

    def _on_rt_taps_computed(self, taps):
        self.rt_last_taps = taps
        self.rt_tap_grid = copy.deepcopy(self.sionna_widget.get_rt_config().get("tap_grid"))
        self.rt_status_label.setText(f"Taps ready: shape {taps.shape}")

    def _ensure_rt_taps(self, fs):
        """Regrid existing paths when the waveform rate or tap controls change."""
        if self.sionna_widget is None or self.rt_last_taps is None:
            return False
        required = {
            "sampling_frequency": float(fs),
            "bandwidth": self.rt_bandwidth_spin.value() * 1e6,
            "l_min": self.rt_l_min_spin.value(), "l_max": self.rt_l_max_spin.value(),
        }
        if self.rt_tap_grid == required:
            return True
        self.rt_last_taps = None
        self.rt_tap_grid = None
        self._rt_push_channel_params()
        taps = self.sionna_widget.compute_taps(sampling_frequency=float(fs))
        if taps is not None:
            self._on_rt_taps_computed(taps)
        if self.rt_tap_grid != required:
            self.rt_last_taps = None
            self.rt_tap_grid = None
            self.rt_status_label.setText("Recompute paths: taps could not be aligned with the waveform grid")
            return False
        return True

    def _on_rt_error(self, msg):
        self.rt_status_label.setText(f"Error: {msg}")
        self.rt_compute_btn.setEnabled(True)
        self.rt_compute_btn.setText("Compute Paths")

    def create_slider_control(self, label, value, unit, min_val, max_val, attr_name):
        """Create a slider control with label and value display"""
        container = QVBoxLayout()
        container.setSpacing(4)

        header = QHBoxLayout()
        label_widget = QLabel(label)
        value_label = QLabel(f"{value} {unit}")
        value_label.setProperty("class", "stat-value")
        header.addWidget(label_widget)
        header.addStretch()
        header.addWidget(value_label)
        container.addLayout(header)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(int(value))

        def update_value(v):
            value_label.setText(f"{v} {unit}")
            setattr(self, attr_name, float(v))

        slider.valueChanged.connect(update_value)
        container.addWidget(slider)

        return container

    def on_dataset_changed(self, name: str):
        if name and name != "No datasets found":
            entry = self.dataset_manager.get_by_name(name)
            if entry:
                self._active_entry = entry
                self.display_original_signal()
                self._update_stochastic_metadata()
                self._update_rt_metadata()
                if self.rt_last_taps is not None:
                    self._ensure_rt_taps(entry['fs'])

    def _update_stochastic_metadata(self):
        """Auto-populate read-only carrier freq / sample rate from dataset metadata."""
        entry = self._active_entry
        if entry is None:
            self.carrier_freq_display.setText("--")
            self.sample_rate_display.setText("--")
            return

        fs = entry['fs']
        fc = entry.get('fc', None)

        self.sample_rate_display.setText(f"{fs / 1e6:.4g}")
        self.carrier_freq_display.setText(f"{float(fc) / 1e6:.4g}" if fc is not None else "N/A")

        # Default output_num_samples to signal length
        sig_len = entry['samples']
        self.output_num_samples = sig_len
        self.output_samples_spin.setValue(sig_len)

    def _update_rt_metadata(self):
        """Auto-populate read-only RT fields from dataset metadata."""
        entry = self._active_entry
        if entry is None:
            self.rt_freq_display.setText("--")
            self.rt_sample_rate_display.setText("--")
            return

        fs = entry['fs']
        fc = entry.get('fc', None)

        self.rt_sample_rate_display.setText(f"{fs / 1e6:.4g}")
        self.rt_freq_display.setText(f"{float(fc) / 1e6:.4g}" if fc is not None else "N/A")

    def upload_dataset(self):
        # Get default directory from settings
        settings = QSettings("MyCompany", "MixedSignalGUI")
        default_dir = settings.value("dataPath", "")
        if not default_dir or not os.path.isdir(default_dir):
            default_dir = ""
        
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import External Dataset", default_dir, "NumPy Files (*.npy);;All Files (*)"
        )
        if not filepath:
            return

        try:
            from PySide6.QtWidgets import QInputDialog
            fs_str, ok = QInputDialog.getText(
                self, "Sample Rate",
                "Enter sample rate (Hz) for this dataset:",
                text="8000000"
            )
            if not ok:
                return
            try:
                fs = float(fs_str)
            except ValueError:
                print("Invalid sample rate — import cancelled.")
                return

            metadata = {
                'source': 'imported',
                'fs':     fs,
            }
            entry = self.dataset_manager.import_external(filepath, metadata)
            self.refresh_dataset_list()
            # Select the newly imported dataset
            self.dataset_combo.setCurrentText(entry['name'])
            print(f"[ChannelTab] Imported: {entry['name']}")

        except Exception as e:
            print(f"Failed to import dataset: {e}")

    def refresh_dataset_list(self):
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        entries = self.dataset_manager.scan()

        if entries:
            names = [e['name'] for e in entries]
            self.dataset_combo.addItems(names)
            # Re-select active entry if it still exists
            if self._active_entry and self._active_entry['name'] in names:
                self.dataset_combo.setCurrentText(self._active_entry['name'])
            else:
                self.dataset_combo.setCurrentIndex(0)
        else:
            self.dataset_combo.addItem("No datasets found")

        self.dataset_combo.blockSignals(False)

        # Trigger a load if we have a valid selection
        current = self.dataset_combo.currentText()
        if current and current != "No datasets found":
            self.on_dataset_changed(current)

        print(f"[ChannelTab] Dataset list refreshed: {self.dataset_combo.count()} entries")

    def display_original_signal(self):
        entry = self._active_entry
        if entry is None:
            print("[ChannelTab] No active dataset to display")
            return

        try:
            self.clean_signal = self.dataset_manager.load_signal(entry)
            self.clean_fs = entry['fs']
            self.clean_metadata = entry   # entire entry is the metadata
        except Exception as e:
            print(f"[ChannelTab] Failed to load signal: {e}")

    # ---- Config builders ----
    def _build_stochastic_config(self):
        entry = self._active_entry or {}
        metadata = entry
        fs = entry.get('fs', 8e6)
        return {
            "seed": experiment_seed(),
            "waveform": {
                "path": None,
                "format": "npy",
                "mat_key": None,
                "sample_rate_hz": fs,
                "normalize_rms": 1.0,
                "iq_layout": "auto",
            },
            "augmentation": {
                "output_num_samples": self.output_num_samples,
                "pad_mode": "repeat",
                "trim_mode": "first",
            },
            "channel": {
                "type": "tdl",
                "profile": self.tdl_profile,
                "delay_spread_s": self.delay_spread_ns * 1e-9,
                "carrier_frequency_hz": float(metadata.get('fc', 1e6)),
                "sample_rate_hz": fs,
                "min_speed_mps": 0.0,
                "max_speed_mps": 0.0,
                "num_tx_ant": 1,
                "num_rx_ant": 1,
                "normalize_channel": False,
                "maximum_delay_spread_s": 3e-6,
                "l_min": None,
                "l_max": None,
            },
            "noise": {
                "snr_db": float(self.stoch_snr_db),
            },
        }

    def _build_rt_config(self):
        """Build multiantenna_config.json-schema dict from SionnaWidget state."""
        self._rt_push_channel_params()
        config = self.sionna_widget.get_rt_config()
        config["seed"] = experiment_seed()
        config["tap_grid"] = copy.deepcopy(self.rt_tap_grid)

        entry = self._active_entry
        if entry:
            fc = entry.get('fc', None)
            if fc is not None:
                config["center_frequency"] = float(fc)
            config["sample_rate"] = entry['fs']

        return config

    def _is_baseband_signal(self, signal):
        """Check if the signal is complex baseband (required for TDL/RT)."""
        return np.iscomplexobj(signal)

    def apply_augmentations(self):
        if not hasattr(self, 'clean_signal'):
            print("No dataset selected to augment.")
            return

        signal = self.clean_signal
        fs = self.clean_fs

        # Stochastic and RT augmentations require complex baseband IQ signals.
        # Real passband signals cast to complex are NOT valid baseband — the
        # carrier frequency relationship is lost and the channel convolution
        # produces physically incorrect results.
        if self.active_subtab in (1, 2) and not self._is_baseband_signal(signal):
            QMessageBox.warning(
                self, "Baseband Signal Required",
                "Stochastic and Ray Tracing channel augmentations operate on "
                "complex baseband (IQ) signals.\n\n"
                "The selected dataset contains a real passband signal, which "
                "cannot be correctly processed by these channel models.\n\n"
                "To fix this, go to the Waveform Generation tab and set the "
                "output type to \"Baseband (Complex IQ)\", then regenerate."
            )
            return

        if self.active_subtab == 1:
            # Stochastic TDL path — sample ranges
            try:
                rng = np.random.default_rng(experiment_seed())
                sampled_delay = self.delay_spread_spin.value_or_range().sample(rng)
                sampled_stoch_snr = self.stoch_snr_spin.value_or_range().sample(rng)

                multi_ch = self.stoch_multi_channel_cb.isChecked()
                config = self._build_stochastic_config()
                # Override with sampled values
                config["channel"]["delay_spread_s"] = sampled_delay * 1e-9
                config["noise"]["snr_db"] = sampled_stoch_snr
                if multi_ch:
                    config["channel"]["num_tx_ant"] = self.stoch_num_tx_ant_spin.value()
                    config["channel"]["num_rx_ant"] = self.stoch_num_rx_ant_spin.value()
                block = StochasticTDLAugmentation(config, multi_channel=multi_ch)
                augmented_signal = block.apply(signal, fs)

                self.last_augmentation_config = {
                    'run_seed': experiment_seed(),
                    'delay_spread_ns': self.delay_spread_spin.value_or_range().to_metadata(sampled_delay),
                    'stoch_snr_db': self.stoch_snr_spin.value_or_range().to_metadata(sampled_stoch_snr),
                    'stoch_config': config,
                }
                if multi_ch and augmented_signal.ndim == 2:
                    self.stoch_output_shape_label.setText(
                        f"Output: {augmented_signal.shape[0]} antennas x {augmented_signal.shape[1]} samples")
                else:
                    self.stoch_output_shape_label.setText("")
            except ImportError as e:
                QMessageBox.warning(
                    self, "Missing Dependencies",
                    f"Stochastic TDL augmentation requires TensorFlow and Sionna.\n\n{e}"
                )
                return
            except Exception as e:
                QMessageBox.critical(
                    self, "Augmentation Error",
                    f"Stochastic augmentation failed:\n\n{e}"
                )
                return
        elif self.active_subtab == 2:
            # Ray Tracing path
            if not self._ensure_rt_taps(fs):
                QMessageBox.warning(
                    self, "No Taps",
                    "Load a scene and compute paths first."
                )
                return
            try:
                multi_ch = self.rt_multi_channel_cb.isChecked()
                config = self._build_rt_config()
                block = SionnaRTAugmentation(config, self.rt_last_taps, multi_channel=multi_ch)
                augmented_signal = block.apply(signal, fs)
                if multi_ch and augmented_signal.ndim == 2:
                    self.rt_output_shape_label.setText(
                        f"Output: {augmented_signal.shape[0]} antennas x {augmented_signal.shape[1]} samples")
                else:
                    self.rt_output_shape_label.setText("")
            except ImportError as e:
                QMessageBox.warning(
                    self, "Missing Dependencies",
                    f"RT augmentation requires TensorFlow and Sionna.\n\n{e}"
                )
                return
            except Exception as e:
                QMessageBox.critical(
                    self, "Augmentation Error",
                    f"RT augmentation failed:\n\n{e}"
                )
                return
        elif self.active_subtab == 3:
            # Measured channel path — a CIR recorded from real hardware.
            # Works on real passband as well as complex baseband, so unlike
            # the TDL and RT paths it is not gated on baseband input.
            block = self._build_measured_block()
            if block is None:
                return                      # reason already shown to the user
            try:
                augmented_signal = block.apply(signal, fs)
                self._last_measured_block = block
            except Exception as e:
                QMessageBox.critical(
                    self, "Augmentation Error",
                    f"Measured channel augmentation failed:\n\n{e}"
                )
                return
        else:
            # AWGN path — sample ranges once per apply
            rng = np.random.default_rng(experiment_seed())
            sampled_snr = self.snr_spin.value_or_range().sample(rng)
            sampled_amp = self.amplitude_spin.value_or_range().sample(rng)
            sampled_phase = self.phase_spin.value_or_range().sample(rng)
            sampled_freq_hz = self.freq_shift_spin.value_or_range().sample(rng)

            pipeline = AugmentationPipeline()
            if self.awgn_enabled:
                pipeline.add(AWGNAugmentation(snr_db=sampled_snr, seed=experiment_seed()))
            if self.amp_phase_enabled:
                phase_rad = np.deg2rad(sampled_phase)
                pipeline.add(ScalarAmplitudeAndPhaseShift(amplitude=sampled_amp, phi=phase_rad))
            if self.freq_shift_enabled:
                pipeline.add(FrequencyShift(delta_f=sampled_freq_hz))
            augmented_signal = pipeline.apply(signal, fs)

            # Record range + sampled values for save_augmented_dataset
            self.last_augmentation_config = {
                'run_seed': experiment_seed(),
                'awgn': {'enabled': self.awgn_enabled,
                         'snr_db': self.snr_spin.value_or_range().to_metadata(sampled_snr)},
                'amp_phase': {'enabled': self.amp_phase_enabled,
                              'amplitude': self.amplitude_spin.value_or_range().to_metadata(sampled_amp),
                              'phase_deg': self.phase_spin.value_or_range().to_metadata(sampled_phase)},
                'freq_shift': {'enabled': self.freq_shift_enabled,
                               'freq_shift_hz': self.freq_shift_spin.value_or_range().to_metadata(sampled_freq_hz)},
            }

        # Get metadata for constellation plot
        metadata = self.clean_metadata
        fc = metadata.get('fc', None)
        Tsymb = metadata.get('Tsymb', None)
        modulation = metadata.get('modulation', None)
        M = metadata.get('M', None)
        alpha = metadata.get('alpha', 0.35)
        span = metadata.get('span', 8)
        pulse_shape = metadata.get('pulse_shape', 'rrc')
        nsymb = metadata.get('Nsymb', None)

        sps = None
        if Tsymb is not None:
            # round(), not int() — see update_waveform_plots(): truncation
            # turns a 0.9999999999999999 product into sps=0, which the
            # constellation path cannot slice with.
            sps = max(1, int(round(fs * Tsymb)))

        # Choose the correct comparison widget
        if self.active_subtab == 2:
            plot_widget = self.rt_comparison_plot
            # Auto-switch to comparison view
            self.rt_view_stack.setCurrentIndex(1)
            self.rt_comparison_pill.setChecked(True)
        else:
            plot_widget = self.comparison_plot

        # Display comparison
        plot_widget.plot_comparison(
            clean_signal=signal,
            augmented_signal=augmented_signal,
            fs=fs,
            fc=fc,
            sps=sps,
            modulation=modulation,
            M=M,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
            nsymb=nsymb,
        )

        # Freeze samples and their provenance together; saving must not read
        # the current dataset, subtab, or controls after this successful apply.
        save_mode = "Single Antenna"
        if self.active_subtab == 1:
            applied_config = self.last_augmentation_config
            if multi_ch:
                save_mode = self.stoch_save_mode_combo.currentText()
        elif self.active_subtab == 2:
            applied_config = copy.deepcopy(config)
            applied_config["waveform_length"] = max(len(signal), int(config.get("waveform_length", len(signal))))
            if multi_ch:
                save_mode = self.rt_save_mode_combo.currentText()
        elif self.active_subtab == 3:
            applied_config = block.to_config()
        else:
            applied_config = self.last_augmentation_config
        self._last_augmentation = {
            "signal": np.array(augmented_signal, copy=True), "fs": fs,
            "source_entry": copy.deepcopy(self.clean_metadata),
            "type": ("awgn", "stochastic_tdl", "sionna_rt", "measured_channel")[self.active_subtab],
            "config": copy.deepcopy(applied_config), "save_mode": save_mode,
        }
        self.last_augmented_signal = self._last_augmentation["signal"]
        self.last_augmented_fs = fs
        self.save_augmented_btn.setEnabled(True)

    def save_augmented_dataset(self):
        if not hasattr(self, '_last_augmentation'):
            print("[ChannelTab] No augmented signal to save. Apply augmentations first.")
            return

        snapshot = self._last_augmentation
        entry = snapshot['source_entry']
        base_name = entry['name'] if entry else "dataset"
        aug_name = self.dataset_manager._unique_name(f"{base_name}_augmented")

        # Build augmentation metadata on top of original entry
        metadata = copy.deepcopy(entry) if entry else {}
        # Drop internal path keys — they will be re-set by save()
        metadata.pop('_npy_path', None)
        metadata.pop('_json_path', None)

        metadata['augmented'] = True
        metadata['base_dataset'] = base_name
        metadata['fs'] = snapshot['fs']
        metadata['source'] = 'augmented'

        metadata['augmentation_type'] = snapshot['type']
        metadata['augmentation_config'] = copy.deepcopy(snapshot['config'])

        # Determine save mode for multi-channel signals
        signal = snapshot['signal']
        save_mode = snapshot['save_mode']

        if save_mode == "All Antennas (Separate Files)" and signal.ndim == 2:
            num_ant = signal.shape[0]
            metadata['num_channels'] = num_ant
            for i in range(num_ant):
                ant_name = self.dataset_manager._unique_name(f"{base_name}_augmented_ant{i}")
                meta_i = dict(metadata)
                meta_i['antenna_index'] = i
                meta_i['channel_group'] = aug_name
                saved = self.dataset_manager.save(ant_name, signal[i], meta_i)
            self._active_entry = saved
            self.refresh_dataset_list()
            print(f"[ChannelTab] Saved {num_ant} per-antenna datasets: {aug_name}_ant*")
        elif save_mode == "All Antennas (Stacked)" and signal.ndim == 2:
            metadata['num_channels'] = signal.shape[0]
            metadata['signal_shape'] = list(signal.shape)
            saved = self.dataset_manager.save(aug_name, signal, metadata)
            self._active_entry = saved
            self.refresh_dataset_list()
            print(f"[ChannelTab] Saved stacked multi-channel dataset: {aug_name} {signal.shape}")
        else:
            # Single antenna or 1D signal — original behavior
            saved = self.dataset_manager.save(aug_name, signal, metadata)
            self._active_entry = saved
            self.refresh_dataset_list()
            print(f"[ChannelTab] Saved augmented dataset: {aug_name}")

    # ------------------------------------------------------------------
    # Bulk augmentation
    # ------------------------------------------------------------------

    def apply_to_all_in_folder(self):
        """Run the current augmentation config over a user-chosen folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Datasets Folder",
            str(self.dataset_manager.datasets_dir))
        if not folder:
            return

        source = DatasetManager(datasets_dir=folder)
        entries = source.scan()
        if not entries:
            QMessageBox.information(self, "Nothing to Process",
                                    "The selected folder contains no datasets.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_dir = os.path.join(folder, f"augmented_{ts}")

        reply = QMessageBox.question(
            self, "Bulk Augmentation",
            f"Apply augmentations to all {len(entries)} datasets?\n\n"
            f"Output folder:\n{dest_dir}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Build ranges + static config from current widget state
        ranges: dict[str, ParameterRange] = {}
        static_config: dict = {}
        rt_taps = None
        measured_channels = None

        if self.active_subtab == 0:
            kind = "awgn"
            ranges["snr_db"] = self.snr_spin.value_or_range()
            ranges["amplitude"] = self.amplitude_spin.value_or_range()
            ranges["phase_deg"] = self.phase_spin.value_or_range()
            ranges["freq_shift_hz"] = self.freq_shift_spin.value_or_range()
            static_config["awgn_enabled"] = self.awgn_enabled
            static_config["amp_phase_enabled"] = self.amp_phase_enabled
            static_config["freq_shift_enabled"] = self.freq_shift_enabled

        elif self.active_subtab == 1:
            kind = "stoch_tdl"
            ranges["delay_spread_ns"] = self.delay_spread_spin.value_or_range()
            ranges["stoch_snr_db"] = self.stoch_snr_spin.value_or_range()
            static_config["stoch_config"] = self._build_stochastic_config()
            multi_ch = self.stoch_multi_channel_cb.isChecked()
            static_config["multi_channel"] = multi_ch
            if multi_ch:
                static_config["stoch_config"]["channel"]["num_tx_ant"] = \
                    self.stoch_num_tx_ant_spin.value()
                static_config["stoch_config"]["channel"]["num_rx_ant"] = \
                    self.stoch_num_rx_ant_spin.value()

        elif self.active_subtab == 2:
            kind = "rt"
            fs = self._active_entry['fs'] if self._active_entry else entries[0].get('fs')
            if fs is None or not self._ensure_rt_taps(fs):
                QMessageBox.warning(self, "No Taps",
                                    "Load a scene and compute paths first.")
                return
            mismatched = [e['name'] for e in entries
                          if not np.isclose(float(e.get('fs', 1.0)), float(fs), rtol=1e-9, atol=0)]
            if mismatched:
                QMessageBox.warning(
                    self, "RT sample-rate mismatch",
                    f"RT taps use {fs:g} Hz. These datasets use a different sample rate:\n"
                    + "\n".join(mismatched)
                    + "\n\nProcess each sample-rate group separately and recompute taps at its rate.")
                return
            rt_taps = self.rt_last_taps
            static_config["rt_config"] = self._build_rt_config()
            multi_ch = self.rt_multi_channel_cb.isChecked()
            static_config["multi_channel"] = multi_ch

        elif self.active_subtab == 3:
            kind = "measured"
            if not len(self.channel_bank):
                QMessageBox.warning(
                    self, "No channels",
                    "Import channel files before applying a measured channel.")
                return
            measured_channels = list(self.channel_bank.channels)
            # With "pick a random channel" ticked, the bulk thread draws one per
            # file from its seeded generator, so a run over N files uses N
            # channels instead of repeating a single realisation N times.
            static_config["random_channel"] = self.meas_random_cb.isChecked()
            static_config["channel_index"] = max(0, self.meas_channel_list.currentRow())
            static_config["normalize"] = self.meas_normalize_cb.isChecked()
            static_config["meas_awgn_enabled"] = self.meas_awgn_cb.isChecked()
            static_config["bank_size"] = len(measured_channels)
            if self.meas_awgn_cb.isChecked():
                # Plain spin box on this page, so the SNR is a fixed value
                # rather than a sweepable range.
                ranges["meas_snr_db"] = ParameterRange(self.meas_snr_spin.value())
        else:
            return

        dest = DatasetManager(datasets_dir=dest_dir)
        self._bulk_thread = BulkAugmentationThread(
            source_manager=source,
            dest_manager=dest,
            augmentation_kind=kind,
            ranges=ranges,
            static_config=static_config,
            rt_taps=rt_taps,
            measured_channels=measured_channels,
            seed=experiment_seed(),
        )

        # Progress dialog
        self._bulk_progress = QProgressDialog(
            "Augmenting datasets…", "Cancel", 0, len(entries), self)
        self._bulk_progress.setWindowTitle("Bulk Augmentation")
        self._bulk_progress.setMinimumDuration(0)
        self._bulk_progress.setModal(True)

        self._bulk_progress.canceled.connect(self._bulk_thread.request_cancel)
        self._bulk_thread.progress.connect(self._on_bulk_progress)
        self._bulk_thread.finished.connect(self._on_bulk_finished)
        self._bulk_thread.error.connect(self._on_bulk_error)

        self._bulk_thread.start()

    def _on_bulk_progress(self, current: int, total: int, name: str):
        self._bulk_progress.setMaximum(total)
        self._bulk_progress.setValue(current)
        self._bulk_progress.setLabelText(f"Processing {current}/{total}: {name}")

    def _on_bulk_finished(self, ok: int, errors: int, dest_dir: str):
        self._bulk_progress.close()
        QMessageBox.information(
            self, "Bulk Augmentation Complete",
            f"Processed: {ok + errors}\n"
            f"Succeeded: {ok}\n"
            f"Errors: {errors}\n\n"
            f"Output: {dest_dir}",
        )
        self.refresh_dataset_list()

    def _on_bulk_error(self, msg: str):
        self._bulk_progress.close()
        QMessageBox.critical(self, "Bulk Augmentation Error", msg)
