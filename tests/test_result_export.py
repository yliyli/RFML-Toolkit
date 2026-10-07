"""Export pairing, repeated runs, failure handling, and real UI result wiring."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication

from mixedsignal_gui.backend.result_export import data_identity, model_identity, export_run, output_folder
from mixedsignal_gui.backend.torch_models import get_model
from mixedsignal_gui.tabs.inference_tab import InferenceResultsTab
from mixedsignal_gui.tabs.evaluate_model_tab import EvaluateModelTab
from mixedsignal_gui.tabs.data_visualization_tab import DataVisualizationTab


class ResultExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'output'
        self.raw = [np.arange(32, dtype=np.float32) + 1j,
                    np.arange(32, dtype=np.float32) - 1j]
        self.labels = ['A', 'B']
        self.checkpoint = self.root / 'cnn.pth'
        metadata = {'model_name': 'TinyConv', 'class_labels': self.labels,
                    'num_classes': 2, 'signal_length': 32, 'input_channels': 2}
        model = get_model('TinyConv', num_classes=2, input_size=32, in_channels=2)
        torch.save(model.state_dict(), self.checkpoint)
        self.checkpoint.with_suffix('.json').write_text(json.dumps(metadata))
        self.model_identity = model_identity(model, metadata, self.checkpoint)

    def test_grouping_repeats_and_changed_data_weights_or_labels(self):
        data = data_identity(self.raw, [0, 1], self.labels)
        fig = Figure()
        fig.subplots().plot([0, 1], [1, 2])
        payload = dict(data=data, model=self.model_identity, kind='batch_evaluation',
                       description='test data', figures={'test': fig},
                       tables={'values': (['value'], [[1], [2]])})
        first = export_run(self.output, **payload)
        second = export_run(self.output, **payload)
        self.assertEqual(first.parent, second.parent)
        self.assertNotEqual(first, second)
        self.assertTrue((first / 'plots/test.png').is_file())
        with (first / 'csv/values.csv').open() as f:
            self.assertEqual(list(csv.reader(f)), [['value'], ['1'], ['2']])
        for changed in (data_identity([self.raw[0] * 2, self.raw[1]], [0, 1], self.labels),
                        data_identity(self.raw, [1, 0], self.labels)):
            run = export_run(self.output, **{**payload, 'data': changed})
            self.assertNotEqual(run.parent, first.parent)
        changed_model = {**self.model_identity, 'sha256': 'changed-weights'}
        run = export_run(self.output, **{**payload, 'model': changed_model})
        self.assertNotEqual(run.parent, first.parent)
        self.assertIn('Runs: 2', (self.output / 'index.txt').read_text())
        self.assertIn(str(self.checkpoint), (first / 'run.txt').read_text())

    def test_failed_export_leaves_no_completed_run_or_index(self):
        figure = MagicMock()
        figure.savefig.side_effect = OSError('disk full')
        with self.assertRaises(OSError):
            export_run(self.output, data=data_identity(self.raw), kind='test',
                       description='test', figures={'bad': figure})
        self.assertEqual(list(self.output.glob('*/run_*')), [])
        self.assertEqual(list(self.output.glob('*/.pending_*')), [])
        self.assertFalse((self.output / 'index.txt').exists())

    def test_batch_button_exports_current_predictions_and_invalidates(self):
        tab = InferenceResultsTab()
        self.addCleanup(tab.close)
        self.assertFalse(tab.export_btn.isEnabled())
        with patch.object(tab, '_device', return_value='cpu'):
            tab._load_model(str(self.checkpoint))
            tab._build_eval_tensors(self.raw, [0, 1])
            tab._evaluate_all()
        self.assertTrue(tab.export_btn.isEnabled())
        with patch('mixedsignal_gui.backend.result_export.output_folder', return_value=self.output), \
             patch('PySide6.QtWidgets.QMessageBox.information'):
            tab.export_btn.click()
            tab.export_btn.click()
        runs = sorted(self.output.glob('*/run_*'))
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].parent, runs[1].parent)
        with (runs[0] / 'csv/predictions.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['true_label'] for r in rows], self.labels)
        self.assertAlmostEqual(float(rows[0]['probability_A']), float(tab._cached_probs[0, 0]))
        self.assertTrue((runs[0] / 'plots/confusion_matrix.png').exists())
        self.assertTrue((runs[0] / 'plots/roc_curves.png').exists())
        tab._build_eval_tensors([self.raw[0]], [0])
        self.assertFalse(tab.export_btn.isEnabled())
        with patch.object(tab, '_device', return_value='cpu'):
            tab._evaluate_all()
        with patch('mixedsignal_gui.backend.result_export.output_folder', return_value=self.output), \
             patch('PySide6.QtWidgets.QMessageBox.information'):
            tab.export_btn.click()
        single_class = next(p for p in self.output.glob('*/run_*') if p.parent != runs[0].parent)
        self.assertFalse((single_class / 'plots/roc_curves.png').exists())
        with patch.object(tab, '_device', return_value='cpu'):
            tab._load_model(str(self.checkpoint))
        self.assertFalse(tab.export_btn.isEnabled())
        self.assertIsNone(tab.eval_data)
        with self.assertRaises(ValueError):
            tab._build_eval_tensors([np.ones(32, dtype=np.float32)], [0])
        self.assertFalse(tab.export_btn.isEnabled())
        self.assertIsNone(tab.eval_data)

    def test_single_waveform_exports_and_model_reload_clears_results(self):
        tab = EvaluateModelTab(None)
        self.addCleanup(tab.close)
        with patch.object(tab, 'get_device', return_value='cpu'):
            tab._do_load_model(str(self.checkpoint))
            tab._last_modulation = 'A'
            tab._plot_signal(self.raw[0], fs=1e6, fc=0, sps=None,
                             modulation=None, m=None, nsymb=None, baseband_symbols=None)
            tab._classify_signal(self.raw[0])
        self.assertTrue(tab.export_btn.isEnabled())
        with patch('mixedsignal_gui.backend.result_export.output_folder', return_value=self.output), \
             patch('PySide6.QtWidgets.QMessageBox.information'):
            tab.export_btn.click()
        run = next(self.output.glob('*/run_*'))
        self.assertTrue((run / 'plots/waveform.png').exists())
        with (run / 'csv/waveform.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 32)
        self.assertEqual(float(rows[0]['imag_0']), 1.)
        tab._last_signal = self.raw[0]
        with patch.object(tab, 'get_device', return_value='cpu'):
            tab._channel_test_classify()
        self.assertTrue(tab.ch_export_btn.isEnabled())
        with patch('mixedsignal_gui.backend.result_export.output_folder', return_value=self.output), \
             patch('PySide6.QtWidgets.QMessageBox.information'):
            tab.ch_export_btn.click()
        channel_run = next(p for p in self.output.glob('*/run_*')
                           if json.loads((p / 'run.json').read_text())['kind'] == 'single_channel_evaluation')
        self.assertTrue((channel_run / 'plots/signals.png').exists())
        with patch.object(tab, 'get_device', return_value='cpu'):
            tab._do_load_model(str(self.checkpoint))
        self.assertFalse(tab.export_btn.isEnabled())
        self.assertFalse(tab.ch_export_btn.isEnabled())

    def test_visualization_exports_coordinates_and_dataset_only_group(self):
        tab = DataVisualizationTab()
        self.addCleanup(tab.close)
        for label, raw in zip(self.labels, self.raw):
            folder = self.root / 'data' / label
            folder.mkdir(parents=True)
            np.save(folder / 'sample.npy', raw)
        tab._data_dir = str(self.root / 'data')
        _, y, classes = tab._collect_features(tab._data_dir, 10)
        tab._class_names = classes
        tab._export_visualization_details = {'method': 'PCA'}
        tab._on_finished(np.array([[0., 1.], [2., 3.]]), np.array([[0., 1., 2.], [3., 4., 5.]]), y)
        with patch('mixedsignal_gui.backend.result_export.output_folder', return_value=self.output), \
             patch('PySide6.QtWidgets.QMessageBox.information'):
            tab.export_btn.click()
        run = next(self.output.glob('visualization_*/run_*'))
        self.assertTrue((run / 'plots/visualization_3d.png').exists())
        with (run / 'csv/coordinates_2d.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r['class'] for r in rows], self.labels)
        self.assertEqual(float(rows[1]['coordinate_2']), 3.)

    def test_model_identity_tracks_loaded_weights_and_metadata(self):
        model = torch.nn.Linear(2, 2)
        original = model_identity(model, {'class_labels': self.labels}, self.checkpoint)
        renamed = model_identity(model, {'class_labels': self.labels}, self.root / 'renamed.pth')
        self.assertEqual(original['sha256'], renamed['sha256'])
        changed_metadata = model_identity(model, {'class_labels': ['B', 'A']}, self.checkpoint)
        self.assertNotEqual(original['sha256'], changed_metadata['sha256'])
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(original['sha256'], model_identity(model, {'class_labels': self.labels}, self.checkpoint)['sha256'])

    def test_output_folder_uses_settings_or_default(self):
        with patch('PySide6.QtCore.QSettings') as settings:
            settings.return_value.value.return_value = str(self.output)
            self.assertEqual(output_folder(), self.output)
            settings.return_value.value.return_value = ''
            self.assertEqual(output_folder(), Path.home() / 'Documents' / 'rfml_output')
