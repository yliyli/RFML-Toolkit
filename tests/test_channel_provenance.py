"""Regression checks for RT grids and apply-time augmentation provenance."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

from mixedsignal_gui.backend.augmentation import MeasuredChannelAugmentation, SionnaRTAugmentation
from mixedsignal_gui.backend.channel_bank import MeasuredChannel
from mixedsignal_gui.backend.dataset_manager import DatasetManager
from mixedsignal_gui.tabs.channel_tab import ChannelNoiseTab


class ChannelProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manager = DatasetManager(str(self.root))
        self.signal = (np.arange(64) + 1j).astype(np.complex64)
        self.manager.save("source", self.signal, {"fs": 8e6, "fc": 3.5e9,
                          "modulation": "QAM", "source": "test", "tag": {"original": True}})
        self.manager.save("other", self.signal * 2, {"fs": 30.72e6, "fc": 3.5e9,
                          "modulation": "PSK", "source": "test"})
        self.tab = ChannelNoiseTab(self.manager)
        self.addCleanup(self.tab.close)
        self.tab.dataset_combo.setCurrentText("source")
        self.tab.comparison_plot.plot_comparison = MagicMock()
        self.tab.rt_comparison_plot.plot_comparison = MagicMock()

    def fake_rt(self):
        widget = MagicMock()
        state = {"tap_grid": {"sampling_frequency": 8e6, "bandwidth": 5e6,
                              "l_min": 0, "l_max": 200},
                 "transmitters": [{"power_dbm": 30}], "waveform_length": 64,
                 "noise_power_dBm": -108}
        widget.get_rt_config.side_effect = lambda: copy.deepcopy(state)
        def compute_taps(sampling_frequency):
            state["tap_grid"] = {"sampling_frequency": sampling_frequency,
                                 "bandwidth": self.tab.rt_bandwidth_spin.value() * 1e6,
                                 "l_min": self.tab.rt_l_min_spin.value(),
                                 "l_max": self.tab.rt_l_max_spin.value()}
            return np.ones((1, 1, 1, 1, 1, 201), dtype=np.complex64)
        widget.compute_taps.side_effect = compute_taps
        self.tab.sionna_widget = widget
        self.tab.rt_last_taps = np.ones((1, 1, 1, 1, 1, 201), dtype=np.complex64)
        self.tab.rt_tap_grid = copy.deepcopy(state["tap_grid"])
        return widget, state

    def saved(self):
        return [e for e in self.manager.scan() if e.get("augmented")]

    def test_save_button_requires_successful_apply(self):
        self.assertEqual(self.tab.save_augmented_btn.text(), "Save Augmented Dataset")
        self.assertFalse(self.tab.save_augmented_btn.isHidden())
        self.assertFalse(self.tab.save_augmented_btn.isEnabled())
        self.tab.save_augmented_btn.click()
        self.assertEqual(self.saved(), [])
        self.tab.apply_augmentations()
        self.assertTrue(self.tab.save_augmented_btn.isEnabled())
        self.tab.save_augmented_btn.click()
        self.assertEqual(len(self.saved()), 1)

    def test_awgn_save_keeps_applied_source_config_and_samples_after_switch(self):
        self.tab.apply_augmentations()
        expected_signal = self.tab.last_augmented_signal.copy()
        expected_config = copy.deepcopy(self.tab.last_augmentation_config)
        self.tab._on_subtab_changed(1)
        self.tab.dataset_combo.setCurrentText("other")
        self.tab.snr_spin.setValue(1)
        self.tab.last_augmentation_config["awgn"]["enabled"] = False
        self.tab.save_augmented_btn.click()
        self.tab.save_augmented_btn.click()  # must not chain onto the first saved output
        entries = self.saved()
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertEqual(entry["base_dataset"], "source")
            self.assertEqual(entry["modulation"], "QAM")
            self.assertEqual(entry["fs"], 8e6)
            self.assertEqual(entry["augmentation_type"], "awgn")
            self.assertEqual(entry["augmentation_config"], expected_config)
            np.testing.assert_array_equal(self.manager.load_signal(entry), expected_signal)

    def test_stochastic_stacked_save_mode_is_captured_at_apply(self):
        self.tab._on_subtab_changed(1)
        self.tab.stoch_multi_channel_cb.setChecked(True)
        self.tab.stoch_save_mode_combo.setCurrentText("All Antennas (Stacked)")
        output = np.stack([self.signal, self.signal * 2])
        with patch("mixedsignal_gui.tabs.channel_tab.StochasticTDLAugmentation") as block:
            block.return_value.apply.return_value = output
            self.tab.apply_augmentations()
        config = copy.deepcopy(self.tab.last_augmentation_config)
        self.tab._on_subtab_changed(0)
        self.tab.stoch_multi_channel_cb.setChecked(False)
        self.tab.last_augmentation_config["stoch_config"]["channel"]["tdl_profile"] = "changed"
        self.tab.save_augmented_btn.click()
        entry = self.saved()[0]
        self.assertEqual(entry["augmentation_type"], "stochastic_tdl")
        self.assertEqual(entry["augmentation_config"], config)
        self.assertEqual(entry["num_channels"], 2)
        np.testing.assert_array_equal(self.manager.load_signal(entry), output)

    def test_rt_separate_antenna_save_keeps_applied_grid_and_config(self):
        widget, state = self.fake_rt()
        self.tab._on_subtab_changed(2)
        self.tab.rt_multi_channel_cb.setChecked(True)
        self.tab.rt_save_mode_combo.setCurrentText("All Antennas (Separate Files)")
        output = np.stack([self.signal, self.signal * 3])
        with patch("mixedsignal_gui.tabs.channel_tab.SionnaRTAugmentation") as block:
            block.return_value.apply.return_value = output
            self.tab.apply_augmentations()
        expected = copy.deepcopy(self.tab._last_augmentation["config"])
        self.tab._on_subtab_changed(0)
        state["transmitters"][0]["power_dbm"] = -10
        self.tab.rt_multi_channel_cb.setChecked(False)
        self.tab.dataset_combo.setCurrentText("other")
        self.tab.save_augmented_btn.click()
        entries = self.saved()
        self.assertEqual(len(entries), 2)
        for entry in entries:
            self.assertEqual(entry["augmentation_type"], "sionna_rt")
            self.assertEqual(entry["augmentation_config"], expected)
            self.assertEqual(entry["base_dataset"], "source")
            self.assertEqual(entry["fs"], 8e6)
            np.testing.assert_array_equal(self.manager.load_signal(entry), output[entry["antenna_index"]])

    def test_measured_save_keeps_channel_used_at_apply(self):
        self.tab._on_subtab_changed(3)
        channel = MeasuredChannel("original-CIR", np.array([1, 0.5j]), fs=8e6)
        block = MeasuredChannelAugmentation(channel)
        with patch.object(self.tab, "_build_measured_block", return_value=block):
            self.tab.apply_augmentations()
        expected = copy.deepcopy(block.to_config())
        channel.name = "changed-CIR"
        self.tab._on_subtab_changed(0)
        self.tab.save_augmented_btn.click()
        self.assertEqual(self.saved()[0]["augmentation_type"], "measured_channel")
        self.assertEqual(self.saved()[0]["augmentation_config"], expected)

    def test_dataset_rate_and_tap_control_changes_recompute_grid(self):
        widget, _ = self.fake_rt()
        self.tab.dataset_combo.setCurrentText("other")
        widget.compute_taps.assert_called_once_with(sampling_frequency=30.72e6)
        self.assertEqual(self.tab.rt_tap_grid["sampling_frequency"], 30.72e6)
        self.tab.rt_bandwidth_spin.setValue(7)
        self.tab.rt_l_max_spin.setValue(300)
        self.assertTrue(self.tab._ensure_rt_taps(30.72e6))
        self.assertEqual(widget.compute_taps.call_count, 2)
        self.assertEqual(self.tab.rt_tap_grid["bandwidth"], 7e6)
        self.assertEqual(self.tab.rt_tap_grid["l_max"], 300)
        self.assertTrue(self.tab._ensure_rt_taps(30.72e6))
        self.assertEqual(widget.compute_taps.call_count, 2)

    def test_failed_regridding_clears_stale_taps_and_blocks_apply(self):
        widget, _ = self.fake_rt()
        widget.compute_taps.side_effect = None
        widget.compute_taps.return_value = None
        self.tab.rt_tap_grid["sampling_frequency"] = 5e6
        self.tab._on_subtab_changed(2)
        with patch("mixedsignal_gui.tabs.channel_tab.QMessageBox.warning") as warning, \
             patch("mixedsignal_gui.tabs.channel_tab.SionnaRTAugmentation") as block:
            self.tab.apply_augmentations()
        warning.assert_called_once()
        block.assert_not_called()
        self.assertIsNone(self.tab.rt_last_taps)
        self.assertFalse(hasattr(self.tab, "_last_augmentation"))
        self.assertFalse(self.tab.save_augmented_btn.isEnabled())

    def test_rt_bulk_rejects_mixed_rates_before_writing(self):
        self.fake_rt()
        self.tab._on_subtab_changed(2)
        with patch("mixedsignal_gui.tabs.channel_tab.QFileDialog.getExistingDirectory", return_value=str(self.root)), \
             patch("mixedsignal_gui.tabs.channel_tab.QMessageBox.question", return_value=QMessageBox.Yes), \
             patch("mixedsignal_gui.tabs.channel_tab.QMessageBox.warning") as warning, \
             patch("mixedsignal_gui.tabs.channel_tab.BulkAugmentationThread") as worker:
            self.tab.apply_to_all_in_folder()
        self.assertIn("other", warning.call_args.args[2])
        worker.assert_not_called()
        self.assertEqual(list(self.root.glob("augmented_*")), [])

    def test_backend_rejects_mismatched_tap_grid_before_loading_sionna(self):
        block = SionnaRTAugmentation({"tap_grid": {"sampling_frequency": 5e6}}, np.ones(1))
        with self.assertRaisesRegex(ValueError, "Recompute taps"):
            block.apply(self.signal, 8e6)


if __name__ == "__main__":
    unittest.main()
