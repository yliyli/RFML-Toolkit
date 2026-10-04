"""Exercise the user-facing Settings and recording-import flows offscreen."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from mixedsignal_gui.backend.dataset_manager import DatasetManager
from mixedsignal_gui.backend.torch_models import TinyConv
from mixedsignal_gui.tabs.ml_training_tab import MLTrainingTab
from mixedsignal_gui.tabs.inference_tab import InferenceResultsTab
from mixedsignal_gui.tabs.evaluate_model_tab import EvaluateModelTab
from mixedsignal_gui.styles.stylesheet import SettingsDialog
from mixedsignal_gui.tabs.data_visualization_tab import DimReduceThread


class PaperInputUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager = DatasetManager(str(self.root / "datasets"))
        self.samples = (np.arange(64) + 1j * np.arange(64)[::-1]).astype(np.complex64)
        self.samples.tofile(self.root / "recording.sigmf-data")
        self.meta_path = self.root / "recording.sigmf-meta"
        self.meta_path.write_text(json.dumps({
            "global": {"core:datatype": "cf32_le", "core:sample_rate": 1e6},
            "captures": [{"core:sample_start": 0, "core:frequency": 900e6}],
            "annotations": [{"core:sample_start": 0, "core:sample_count": 16, "core:label": "QAM"},
                            {"core:sample_start": 32, "core:sample_count": 16, "core:label": "PSK"}],
        }))

    def test_training_import_button_and_registry_classes(self):
        tab = MLTrainingTab(self.manager)
        self.addCleanup(tab.close)
        with patch("mixedsignal_gui.tabs.ml_training_tab.QFileDialog.getOpenFileName",
                   return_value=(str(self.meta_path), "")):
            tab.import_sigmf_btn.click()
        self.assertEqual(len(self.manager.scan()), 2)
        tab.load_from_registry()
        self.assertEqual(set(tab.datasets), {"QAM", "PSK"})
        for label, files in tab.datasets.items():
            self.assertEqual(len(files), 1)
            self.assertTrue(np.iscomplexobj(np.load(files[0])))

    def test_visualization_uses_settings_seed(self):
        with patch("mixedsignal_gui.tabs.data_visualization_tab.experiment_seed", return_value=137):
            worker = DimReduceThread(np.ones((8, 4)), [0] * 8, "PCA", {})
        with patch("sklearn.decomposition.PCA") as pca:
            worker.run()
        self.assertEqual(len(pca.call_args_list), 2)
        for call in pca.call_args_list:
            self.assertEqual(call.kwargs["random_state"], 137)

    def test_evaluate_import_selects_region_and_uses_capture_metadata(self):
        tab = EvaluateModelTab(None, self.manager)
        self.addCleanup(tab.close)
        tab.model = TinyConv(num_classes=2, in_channels=2)
        tab.model_metadata = {"input_channels": 2, "signal_length": 32}
        tab.class_labels = ["QAM", "PSK"]
        with patch("mixedsignal_gui.tabs.evaluate_model_tab.QFileDialog.getOpenFileName",
                   return_value=(str(self.meta_path), "")), \
             patch("mixedsignal_gui.tabs.evaluate_model_tab.QInputDialog.getItem",
                   side_effect=lambda *args: (args[3][1], True)), \
             patch.object(tab, "_plot_signal") as plot:
            tab.import_and_classify()
        np.testing.assert_array_equal(tab._last_signal, self.samples[32:48])
        self.assertEqual(tab._last_modulation, "PSK")
        self.assertEqual(plot.call_args.kwargs["fs"], 1e6)
        self.assertEqual(plot.call_args.kwargs["fc"], 900e6)
        self.assertIn("%", tab.result_label.text())
        self.assertEqual(self.manager.scan()[0]["samples"], 16)

    def test_inference_import_and_registry_preparation(self):
        tab = InferenceResultsTab(self.manager)
        self.addCleanup(tab.close)
        tab.model = TinyConv(num_classes=2, in_channels=2)
        tab.model_in_channels = 2
        tab.model_signal_length = 32
        tab.class_labels = ["QAM", "PSK"]
        with patch("mixedsignal_gui.tabs.inference_tab.QFileDialog.getOpenFileName",
                   return_value=(str(self.meta_path), "")):
            tab.import_sigmf_btn.click()
        tab._on_load_from_registry()
        self.assertEqual(tuple(tab.eval_data.shape), (2, 2, 32))
        self.assertEqual(set(tab.eval_labels), {0, 1})
        self.assertTrue(tab.eval_all_btn.isEnabled())

    def test_settings_seed_is_saved_and_loaded_again(self):
        settings = QSettings(str(self.root / "settings.ini"), QSettings.IniFormat)
        with patch("mixedsignal_gui.styles.stylesheet.QSettings", return_value=settings):
            first = SettingsDialog()
            self.addCleanup(first.close)
            first.experiment_seed_spin.setValue(137)
            first._save_values()
            settings.sync()
            second = SettingsDialog()
            self.addCleanup(second.close)
            self.assertEqual(second.experiment_seed_spin.value(), 137)


if __name__ == "__main__":
    unittest.main()
