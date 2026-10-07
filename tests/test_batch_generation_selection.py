"""Batch selection controls keep checkboxes and generation config in sync."""
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import numpy as np

from PySide6.QtWidgets import QApplication, QDialog

from mixedsignal_gui.tabs.waveform_tab import BatchGenerationConfigDialog, WaveformSelectionTab


class BatchGenerationSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        with patch("mixedsignal_gui.tabs.waveform_tab.unavailable_modulations",
                   return_value={"5G_NR": "5G Toolbox"}):
            self.dialog = BatchGenerationConfigDialog({"num_samples": 2})
        self.addCleanup(self.dialog.close)

    def test_initial_selection_matches_config(self):
        self.assertEqual(self.dialog.get_config()['_output_type'], 'baseband')
        for mod, checkbox in self.dialog.mod_checkboxes.items():
            self.assertEqual(checkbox.isChecked(), self.dialog.config[mod]['enabled'])
        self.assertFalse(self.dialog.mod_checkboxes['5G_NR'].isChecked())

    def test_bulk_selection_preserves_parameters_and_manual_selection(self):
        before = self.dialog.get_config()
        self.dialog.unselect_all_btn.click()
        self.assertTrue(all(not c.isChecked() for c in self.dialog.mod_checkboxes.values()))
        self.assertTrue(all(not self.dialog.config[m]['enabled'] for m in self.dialog.modulations))
        self.dialog.mod_checkboxes['LoRa'].click()
        self.assertEqual([m for m in self.dialog.modulations if self.dialog.config[m]['enabled']],
                         ['LoRa'])
        self.dialog.select_all_btn.click()
        self.assertTrue(all(c.isChecked() for c in self.dialog.mod_checkboxes.values()))
        self.assertTrue(all(self.dialog.config[m]['enabled'] for m in self.dialog.modulations))
        after = self.dialog.get_config()
        self.assertEqual(before['_output_type'], after['_output_type'])
        for mod in self.dialog.modulations:
            self.assertEqual({k: v for k, v in before[mod].items() if k != 'enabled'},
                             {k: v for k, v in after[mod].items() if k != 'enabled'})

    def test_batch_generation_excludes_output_metadata_and_honors_selector(self):
        self.dialog.unselect_all_btn.click()
        self.dialog.mod_checkboxes['LoRa'].setChecked(True)
        tab = SimpleNamespace(fs=8e6, fc=1e6, Tsymb=1e-6, var=1., alpha=.35,
                              span=10, Nsymb=256, output_type='baseband', matlab=None,
                              batch_samples_spin=MagicMock(), pulse_shape_combo=MagicMock(),
                              dataset_manager=MagicMock())
        tab._generation_destination = lambda: tab.dataset_manager
        tab.batch_samples_spin.value.return_value = 2
        tab.pulse_shape_combo.currentText.return_value = 'rrc'
        for index, expected in ((0, 'passband'), (1, 'baseband')):
            with self.subTest(output_type=expected):
                self.dialog._output_type_combo.setCurrentIndex(index)
                tab.dataset_manager.reset_mock()
                with patch('mixedsignal_gui.tabs.waveform_tab.BatchGenerationConfigDialog') as dialog_cls, \
                     patch('mixedsignal_gui.backend.waveform_pipeline.WaveformPipeline') as pipeline_cls, \
                     patch('mixedsignal_gui.tabs.waveform_tab.QMessageBox.information') as info:
                    dialog_cls.return_value.exec.return_value = QDialog.Accepted
                    dialog_cls.return_value.get_config.return_value = self.dialog.get_config()
                    pipeline_cls.return_value.generate.return_value = {'signal': np.ones(2048)}
                    WaveformSelectionTab.batch_generate(tab)
                    self.assertEqual(pipeline_cls.return_value.generate.call_count, 2)
                    for call in pipeline_cls.return_value.generate.call_args_list:
                        self.assertEqual(call.kwargs['output_type'], expected)
                    self.assertEqual(tab.dataset_manager.save.call_count, 2)
                    for call in tab.dataset_manager.save.call_args_list:
                        self.assertEqual(call.args[2]['output_type'], expected)
                    self.assertIn('Saved 2 of 2', info.call_args.args[2])
