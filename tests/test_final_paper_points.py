"""Python LoRa generation and consistent model-input normalization."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton

from mixedsignal_gui.backend.core import Waveform, WaveformConfig
from mixedsignal_gui.backend.generators import PythonWaveformGenerator
from mixedsignal_gui.backend.dataset_manager import DatasetManager
from mixedsignal_gui.backend.preprocessing import POWER_NORMALIZATION, normalize_model_input
from mixedsignal_gui.backend.trainer import TrainerThread, pack_multichannel
from mixedsignal_gui.tabs.evaluate_model_tab import EvaluateModelTab
from mixedsignal_gui.tabs.inference_tab import InferenceResultsTab
from mixedsignal_gui.tabs.waveform_tab import WaveformSelectionTab


class FinalPaperPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def config(self, **kwargs):
        values = dict(modulation="LoRa", fs=1e6, Tsymb=2e-6, fc=1e5, M=7, Nsymb=256)
        values.update(kwargs)
        return WaveformConfig(**values)

    def test_sample_length_default_and_validation(self):
        self.assertEqual(TrainerThread([], ['A', 'B']).TARGET_LENGTH, 2048)
        self.assertEqual(TrainerThread([], ['A', 'B'], sample_length=4096).TARGET_LENGTH, 4096)
        with self.assertRaises(ValueError):
            TrainerThread([], ['A', 'B'], sample_length=0)

    def test_lora_lengths_power_seed_and_spreading_factor_range(self):
        for sf in range(7, 13):
            cfg = self.config(M=sf, Tsymb=3.5e-6)
            first = PythonWaveformGenerator(seed=42)
            x = first.generate(cfg)
            self.assertEqual(len(x), cfg.output_len)
            self.assertTrue(np.iscomplexobj(x))
            np.testing.assert_allclose(np.abs(x), 1, atol=1e-12)
            np.testing.assert_array_equal(x, PythonWaveformGenerator(seed=42).generate(cfg))
            self.assertEqual(first.last_metadata["lora"]["spreading_factor"], sf)
            self.assertEqual(first.last_metadata["lora"]["oversampling"], 4)
        for requested, expected in ((4, 7), (99, 12)):
            generator = PythonWaveformGenerator(42)
            generator.generate(self.config(M=requested))
            self.assertEqual(generator.last_metadata["lora"]["spreading_factor"], expected)
        self.assertFalse(np.array_equal(PythonWaveformGenerator(42).generate(self.config()),
                                       PythonWaveformGenerator(43).generate(self.config())))

    def test_lora_analytic_phase_wrap_downchirp_and_quarter_boundary(self):
        make = PythonWaveformGenerator._lora_chirp
        base, _ = make(0, 128, 256, 5e5, 1e6, 0)
        shifted, _ = make(37, 128, 256, 5e5, 1e6, 0)
        down, _ = make(0, 128, 256, 5e5, 1e6, 0, down=True)
        np.testing.assert_allclose(down, np.conj(base))
        self.assertEqual(np.argmax(np.abs(np.fft.fft((shifted * np.conj(base))[::2]))), 37)
        quarter, phase = make(0, 128, 256, 5e5, 1e6, 0, down=True, cut=64)
        following, _ = make(0, 128, 256, 5e5, 1e6, phase)
        np.testing.assert_allclose(quarter, down[:64])
        np.testing.assert_allclose(following[0], down[64])

    def test_lora_passband_matches_quadrature_upconversion(self):
        cfg = self.config()
        bb = PythonWaveformGenerator(42).generate(cfg)
        real = PythonWaveformGenerator(42).generate(self.config(output_type="passband"))
        self.assertFalse(np.iscomplexobj(real))
        np.testing.assert_allclose(real, np.real(bb * np.exp(2j * np.pi * cfg.fc * np.arange(len(bb)) / cfg.fs)), atol=1e-12)

    def test_lora_dispatch_never_calls_live_matlab(self):
        engine = MagicMock()
        engine.is_available.return_value = True
        waveform = Waveform(fs=1e6, Tsymb=2e-6, Nsymb=64, fc=1e5, M=7,
                            modulation="LoRa", matlab_engine=engine, seed=42)
        waveform.generate()
        self.assertEqual(waveform.metadata["generator"], "python")
        engine.eng.waveform_generator.assert_not_called()

    def test_lora_gui_generate_and_save_without_matlab(self):
        manager = DatasetManager(str(self.root / "data"))
        tab = WaveformSelectionTab(None, manager)
        self.addCleanup(tab.close)
        tab.waveform_combo.setCurrentText("LoRa")
        tab.fs, tab.Tsymb, tab.fc, tab.M, tab.Nsymb = 1e6, 2e-6, 1e5, 7, 256
        with patch.object(tab, "update_waveform_plots"), \
             patch("mixedsignal_gui.tabs.waveform_tab.QMessageBox.critical") as error, \
             patch("mixedsignal_gui.tabs.waveform_tab.QMessageBox.warning") as warning:
            buttons = tab.findChildren(QPushButton)
            next(b for b in buttons if "Generate a Sample" in b.text()).click()
            next(b for b in buttons if "Save Sample to Dataset" in b.text()).click()
        error.assert_not_called()
        warning.assert_not_called()
        self.assertEqual(manager.scan(), [])
        saved = DatasetManager(manager.datasets_dir / tab.dataset_name_edit.text()).scan()
        self.assertEqual(saved[0]["generator"], "python")
        self.assertEqual(saved[0]["modulation"], "LoRa")

    def test_normalization_all_layouts_scale_invariance_and_zero(self):
        for iq, channels in ((False, 1), (True, 2), (False, 3), (True, 6)):
            x = np.arange(1, channels * 16 + 1, dtype=np.float32).reshape(1, channels, 16)
            normalized = normalize_model_input(x, complex_iq=iq)
            np.testing.assert_allclose(normalized, normalize_model_input(x * 1e-15, complex_iq=iq), atol=2e-7)
            antennas = channels // 2 if iq else channels
            self.assertAlmostEqual(float(np.mean(np.sum(normalized ** 2, axis=1)) / antennas), 1, places=6)
            np.testing.assert_array_equal(normalize_model_input(np.zeros_like(x), complex_iq=iq), 0)
            if channels > 1:
                np.testing.assert_allclose(normalized[0, -1] / normalized[0, 0], x[0, -1] / x[0, 0], rtol=1e-6)

    def test_training_and_both_inference_views_share_normalization(self):
        for layout in ("real", "iq", "multi_real", "multi_iq"):
            raw = np.arange(1, 17, dtype=np.float32)
            iq = "iq" in layout
            if iq:
                raw = raw + 1j * raw[::-1]
            if layout.startswith("multi"):
                raw = np.stack([raw, raw * 3])
            files = []
            for i, scale in enumerate((1, 10, .001, 2)):
                path = self.root / f"{layout}{i}.npy"
                np.save(path, raw * scale)
                files.append((str(path), i % 2))
            worker = TrainerThread(files, ["A", "B"], model_name="TinyConv", epochs=1,
                                   batch_size=2, save_dir=str(self.root / layout), seed=42,
                                   sample_length=32)
            paths = []
            worker.finished.connect(paths.append)
            with patch("mixedsignal_gui.backend.trainer.normalize_model_input", wraps=normalize_model_input) as normalizer:
                worker.run()
            self.assertTrue(paths[0], worker.error)
            self.assertEqual(normalizer.call_count, 1)
            meta = json.loads(Path(paths[0]).with_suffix(".json").read_text())
            self.assertEqual(meta["power_normalization"], {"method": POWER_NORMALIZATION, "complex_iq": iq})
            expected = normalize_model_input(pack_multichannel(raw, 32)[None], complex_iq=iq)
            np.testing.assert_allclose(normalize_model_input(normalizer.call_args.args[0], complex_iq=iq),
                                       np.repeat(expected, 4, axis=0), atol=2e-7)
            batch = InferenceResultsTab()
            self.addCleanup(batch.close)
            with patch.object(batch, "_device", return_value="cpu"):
                batch._load_model(paths[0])
            self.assertIsNotNone(batch.model)
            self.assertEqual(batch.sample_length_label.text(), "Sample Length: 32 (checkpoint)")
            self.assertEqual(batch.model_metadata["power_normalization"], meta["power_normalization"])
            batch._build_eval_tensors([raw, raw * 10], [0, 1])
            np.testing.assert_allclose(batch.eval_data.numpy(), np.repeat(expected, 2, axis=0), atol=2e-7)
            single = EvaluateModelTab(None)
            self.addCleanup(single.close)
            with patch.object(single, "get_device", return_value="cpu"):
                single._do_load_model(paths[0])
            self.assertIsNotNone(single.model)
            self.assertEqual(single.sample_length_label.text(), "Sample Length: 32 (checkpoint)")
            received = []
            hook = single.model.register_forward_pre_hook(lambda _, args: received.append(args[0].cpu().numpy().copy()))
            self.addCleanup(hook.remove)
            single._last_modulation = layout
            with patch.object(single, "get_device", return_value="cpu"), patch.object(single, "_plot_probabilities"):
                single._classify_signal(raw)
                prediction = single.result_label.text()
                single._classify_signal(raw * 10)
            np.testing.assert_allclose(received[0], expected, atol=2e-7)
            np.testing.assert_allclose(received[1], expected, atol=2e-7)
            self.assertEqual(single.result_label.text(), prediction)

    def test_legacy_checkpoint_preprocessing_remains_unchanged(self):
        batch = InferenceResultsTab()
        self.addCleanup(batch.close)
        batch.model_in_channels = 1
        batch.model_signal_length = 16
        raw = np.arange(16, dtype=np.float32)
        batch._build_eval_tensors([raw], [0])
        np.testing.assert_array_equal(batch.eval_data.numpy()[0, 0], raw)


if __name__ == "__main__":
    unittest.main()
