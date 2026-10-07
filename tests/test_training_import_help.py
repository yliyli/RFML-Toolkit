"""Exercise the consolidated import buttons and non-modal help offscreen."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QPushButton

from mixedsignal_gui.backend.dataset_manager import DatasetManager
from mixedsignal_gui.tabs.ml_training_tab import MLTrainingTab
from mixedsignal_gui.widgets.hover_help import HoverHelpButton


class TrainingImportHelpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tab = MLTrainingTab(DatasetManager(str(self.root / "registry")))
        self.addCleanup(self.tab.close)

    def test_entry_points_and_guidance(self):
        self.assertEqual(self.tab.load_registry_btn.text(), "Load Toolbox Datasets")
        self.assertEqual(self.tab.add_data_btn.text(), "Import External Data")
        self.assertEqual(self.tab.import_sigmf_btn.text(), "Import Long SigMF")
        self.assertFalse(hasattr(self.tab, "quick_load_dataset"))
        self.assertFalse(any("Quick Load" in b.text() for b in self.tab.findChildren(QPushButton)))
        layout = self.tab.load_registry_btn.parentWidget().layout()
        row = next(layout.itemAt(i).layout() for i in range(layout.count())
                   if layout.itemAt(i).layout() is not None
                   and layout.itemAt(i).layout().indexOf(self.tab.load_registry_btn) >= 0)
        self.assertEqual([row.itemAt(i).widget() for i in range(3)],
                         [self.tab.load_registry_btn, self.tab.add_data_btn, self.tab.import_sigmf_btn])
        for expected in ("Settings", "core:sample_start", "core:sample_count", "core:label",
                         "no automatic windowing", "(antennas, N)"):
            self.assertIn(expected, self.tab.dataset_help_btn.help_label.text())
        self.assertIn("build_model", self.tab.model_plugin_help_btn.help_label.text())

    def test_external_import_class_subfolders(self):
        external = self.root / "external"
        for label in ("LoRa", "BPSK"):
            folder = external / label
            folder.mkdir(parents=True)
            np.save(folder / "arbitrary.npy", np.ones(2048, dtype=np.complex64))
        with patch("mixedsignal_gui.tabs.ml_training_tab.QFileDialog.getExistingDirectory",
                   return_value=str(external)):
            self.tab.add_data_btn.click()
        self.assertEqual(set(self.tab.datasets), {"LoRa", "BPSK"})
        self.assertTrue(self.tab.train_btn.isEnabled())

    def test_flat_folder_is_rejected_with_guidance(self):
        np.save(self.root / "flat.npy", np.ones(2048))
        with patch("mixedsignal_gui.tabs.ml_training_tab.QFileDialog.getExistingDirectory",
                   return_value=str(self.root)):
            self.tab.add_data_btn.click()
        self.assertEqual(self.tab.datasets, {})
        self.assertIn("class subfolders", self.tab.status_label.text())

    def test_toolbox_picker_merges_folders_and_deduplicates(self):
        folders = [self.root / "first", self.root / "second"]
        for folder in folders:
            manager = DatasetManager(str(folder))
            for label in ("LoRa", "PSK"):
                manager.save(label, np.ones(2048, dtype=np.complex64),
                             {"source": "test", "fs": 30.72e6, "modulation": label})
            manager.save("held_out", np.ones(2048),
                         {"source": "test", "fs": 30.72e6,
                          "modulation": "LoRa", "data_split": "test"})
        with patch("mixedsignal_gui.tabs.ml_training_tab.QFileDialog.getExistingDirectory",
                   side_effect=[str(folders[0]), str(folders[1]), str(folders[0]), ""]) as picker:
            for _ in range(4):
                self.tab.load_registry_btn.click()
        self.assertEqual(set(self.tab.datasets), {"LoRa", "PSK"})
        self.assertEqual([len(v) for v in self.tab.datasets.values()], [2, 2])
        self.assertEqual(self.tab.dataset_list.count(), 2)
        self.assertTrue(self.tab.train_btn.isEnabled())
        self.assertEqual(self.tab.dataset_manager.datasets_dir, self.root / "registry")
        self.assertEqual(picker.call_args.args[2], str(self.root / "registry"))

    def test_one_class_folders_can_be_loaded_incrementally(self):
        for label in ("LoRa", "PSK"):
            manager = DatasetManager(str(self.root / label))
            manager.save("example", np.ones(2048),
                         {"source": "test", "fs": 1e6, "modulation": label})
            self.tab.load_from_registry(str(manager.datasets_dir))
            self.assertEqual(self.tab.train_btn.isEnabled(), label == "PSK")
        self.assertEqual(set(self.tab.datasets), {"LoRa", "PSK"})

    def test_hover_panel_lifecycle(self):
        button = HoverHelpButton("Example help")
        self.addCleanup(button.close)
        button.show()
        button.show_help()
        self.app.processEvents()
        self.assertTrue(button.popup.isVisible())
        self.app.sendEvent(button, QEvent(QEvent.Leave))
        self.assertTrue(button.hide_timer.isActive())
        self.app.sendEvent(button.help_label, QEvent(QEvent.Enter))
        self.assertFalse(button.hide_timer.isActive())
        self.assertTrue(button.popup.isVisible())
        with patch.object(button, "underMouse", return_value=False), \
             patch.object(button.popup, "underMouse", return_value=True):
            button._hide_if_outside()
            self.assertTrue(button.popup.isVisible())
        with patch.object(button, "underMouse", return_value=False), \
             patch.object(button.popup, "underMouse", return_value=False):
            self.app.sendEvent(button.popup, QEvent(QEvent.Leave))
            self.assertTrue(button.hide_timer.isActive())
            # Offscreen platforms can synthesize pointer-enter events at the
            # virtual cursor. Emit the timeout to test dismissal deterministically.
            button.hide_timer.timeout.emit()
            self.assertFalse(button.popup.isVisible())
        button.click()
        self.assertTrue(button.popup.isVisible())
        button.hide()
        self.assertFalse(button.popup.isVisible())


if __name__ == "__main__":
    unittest.main()
