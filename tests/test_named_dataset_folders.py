"""Named dataset folders, safe names, and discovery from channel controls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from mixedsignal_gui.backend.dataset_manager import DatasetManager, dataset_folder_path
from mixedsignal_gui.tabs.waveform_tab import WaveformSelectionTab, BatchGenerationConfigDialog
from mixedsignal_gui.tabs.channel_tab import ChannelNoiseTab


class NamedDatasetFolderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.manager = DatasetManager(self.root)

    def test_names_cannot_escape_root(self):
        for name in ('', '.', '..', '../outside', '/absolute', 'a/b', 'a\\b', 'a:b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                dataset_folder_path(self.root, name)
        self.assertEqual(dataset_folder_path(self.root, ' my dataset '), self.root / 'my dataset')
        (self.root / 'existing').write_text('keep')
        with self.assertRaises(ValueError):
            dataset_folder_path(self.root, 'existing')

    def test_batch_saves_only_inside_named_folder_and_channel_discovers_it(self):
        tab = WaveformSelectionTab(None, self.manager)
        self.addCleanup(tab.close)
        tab.dataset_name_edit.setText('demo_raw')
        dialog = BatchGenerationConfigDialog({'num_samples': 2})
        self.addCleanup(dialog.close)
        dialog.unselect_all_btn.click()
        dialog.mod_checkboxes['LoRa'].setChecked(True)
        with patch('mixedsignal_gui.tabs.waveform_tab.BatchGenerationConfigDialog') as cls, \
             patch('mixedsignal_gui.backend.waveform_pipeline.WaveformPipeline') as pipeline, \
             patch('mixedsignal_gui.tabs.waveform_tab.QMessageBox.information'):
            cls.return_value.exec.return_value = QDialog.Accepted
            cls.return_value.get_config.return_value = dialog.get_config()
            pipeline.return_value.generate.return_value = {'signal': np.ones(32, dtype=np.complex64)}
            tab.batch_generate()
        self.assertEqual(self.manager.scan(), [])
        self.assertEqual(len(DatasetManager(self.root / 'demo_raw').scan()), 2)
        channel = ChannelNoiseTab(self.manager)
        self.addCleanup(channel.close)
        self.assertEqual(channel.dataset_combo.count(), 2)
        self.assertTrue(channel.dataset_combo.currentText().startswith('demo_raw/'))
        self.assertEqual(channel.clean_signal.shape, (32,))

    def test_bulk_custom_name_and_existing_folder_are_safe(self):
        source = DatasetManager(self.root / 'raw')
        source.save('LoRa', np.ones(32, dtype=np.complex64),
                    {'source': 'test', 'fs': 1e6, 'modulation': 'LoRa'})
        tab = ChannelNoiseTab(self.manager)
        self.addCleanup(tab.close)
        tab.augmentation_name_edit.setText('demo_awgn')
        with patch('mixedsignal_gui.tabs.channel_tab.QFileDialog.getExistingDirectory', return_value=str(source.datasets_dir)), \
             patch('mixedsignal_gui.tabs.channel_tab.QMessageBox.question', return_value=QMessageBox.Yes), \
             patch('mixedsignal_gui.tabs.channel_tab.BulkAugmentationThread') as worker:
            tab.apply_to_all_in_folder()
            self.assertEqual(worker.call_args.kwargs['dest_manager'].datasets_dir,
                             self.root / 'demo_awgn')
            self.assertFalse((self.root / 'raw' / 'demo_awgn').exists())
            tab._bulk_progress.close()
        with patch('mixedsignal_gui.tabs.channel_tab.QFileDialog.getExistingDirectory', return_value=str(source.datasets_dir)), \
             patch('mixedsignal_gui.tabs.channel_tab.QMessageBox.warning') as warning, \
             patch('mixedsignal_gui.tabs.channel_tab.BulkAugmentationThread') as worker:
            tab.apply_to_all_in_folder()
            worker.assert_not_called()
            self.assertIn('already exists', warning.call_args.args[2])
