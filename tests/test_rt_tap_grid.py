"""Numerical RT lag-grid checks; Sionna dependencies are optional."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("sionna") and importlib.util.find_spec("tensorflow"),
                     "Sionna/TensorFlow are not installed")
class RTTapGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mixedsignal_gui.sionna_widget.engine import SimpleSimulationEngine
        from mixedsignal_gui.sionna_widget.widget import SionnaWidget
        cls.engine_class = SimpleSimulationEngine
        cls.widget_class = SionnaWidget

    def test_scene_switch_does_not_install_native_busy_cursor(self):
        harness = SimpleNamespace(
            _computing=True, _viewport=MagicMock(), _engine=MagicMock(),
            _controls=None, _sync_viewport_markers=MagicMock(),
            scene_loaded=MagicMock(), error_occurred=MagicMock())
        with patch("mixedsignal_gui.sionna_widget.widget.QApplication") as app:
            for success in (True, False):
                harness._engine.load_scene.return_value = success
                self.widget_class.load_scene(harness, "scene.xml")
                self.assertFalse(harness._computing)
            app.setOverrideCursor.assert_not_called()
            app.restoreOverrideCursor.assert_not_called()
        harness.scene_loaded.emit.assert_called_once_with("scene.xml")
        harness.error_occurred.emit.assert_called_once_with("Failed to load scene: scene.xml")

    def test_known_delay_stays_fixed_when_sample_rate_changes(self):
        engine = self.engine_class()
        paths = MagicMock()
        paths.cir.return_value = (np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex64),
                                 np.array([[[1e-6]]]))
        engine._last_paths = paths
        for fs in (5e6, 8e6, 30.72e6):
            taps = engine.compute_taps(bandwidth=5e6, sampling_frequency=fs, l_min=0, l_max=64)
            self.assertEqual(taps.shape, (1, 1, 1, 1, 1, 65))
            peak = np.argmax(np.abs(taps.reshape(-1)))
            self.assertAlmostEqual(peak / fs, 1e-6, delta=0.5 / fs)
            expected = (5e6 / fs) * np.sinc(5e6 * (np.arange(65) / fs - 1e-6))
            np.testing.assert_allclose(taps.reshape(-1), expected, atol=1e-7)
            self.assertEqual(engine.get_full_config()["tap_grid"]["sampling_frequency"], fs)
            self.assertEqual(engine.get_full_config()["sample_rate"], fs)

    def test_explicit_antenna_delays_and_empty_paths(self):
        engine = self.engine_class()
        paths = MagicMock()
        coefficients = np.ones((1, 2, 1, 1, 1, 1), dtype=np.complex64)
        delays = np.array([[[[[1e-6]]], [[[2e-6]]]]])
        paths.cir.return_value = coefficients, delays
        engine._last_paths = paths
        taps = engine.compute_taps(5e6, 8e6, 0, 64)
        self.assertEqual(np.argmax(np.abs(taps[0, 0])), 8)
        self.assertEqual(np.argmax(np.abs(taps[0, 1])), 16)
        paths.cir.return_value = coefficients[..., :0, :], delays[..., :0]
        taps = engine.compute_taps(5e6, 8e6, 0, 64)
        self.assertEqual(taps.shape, (1, 2, 1, 1, 1, 65))
        self.assertTrue(np.all(taps == 0))

    def test_widget_automatic_taps_use_dataset_rate_not_bandwidth(self):
        engine = MagicMock()
        harness = SimpleNamespace(
            _get_channel_params_dict=lambda: {"bandwidth_hz": 5e6, "sample_rate_hz": 30.72e6,
                                             "l_min": 0, "l_max": 200},
            _engine=engine, taps_computed=MagicMock())
        self.widget_class.compute_taps(harness)
        engine.compute_taps.assert_called_once_with(bandwidth=5e6, sampling_frequency=30.72e6,
                                                   l_min=0, l_max=200)

    def test_real_sionna_los_paths_and_convolution_at_two_rates(self):
        # Dr.Jit and offscreen Qt can race during interpreter teardown. Keep
        # the real native ray-tracing integration in a fresh, non-GUI process.
        result = subprocess.run([sys.executable, __file__, "--real-sionna"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def check_real_sionna(self):
        from mixedsignal_gui.backend.augmentation import SionnaRTAugmentation
        engine = self.engine_class()
        engine.set_tx_position([0, 0, 1])
        engine.set_rx_position([299.792458, 0, 1])  # one microsecond propagation
        engine.set_antenna_arrays({}, {"num_cols": 2})
        engine.set_solver_options({"max_depth": 0, "samples_per_src": 100,
                                   "diffuse_reflection": False, "specular_reflection": False,
                                   "refraction": False})
        self.assertTrue(engine.load_scene(None))  # empty geometry, direct path only
        self.assertIsNotNone(engine.compute_paths())
        native = engine.last_paths.taps(bandwidth=5e6, sampling_frequency=5e6,
                                       l_min=0, l_max=64, normalize_delays=False, out_type="numpy")
        np.testing.assert_allclose(engine.compute_taps(5e6, 5e6, 0, 64), native, atol=1e-9)
        signal = np.zeros(128, dtype=np.complex64)
        signal[0] = 1
        for fs, l_min in ((8e6, 0), (30.72e6, 0), (8e6, 4)):
            taps = engine.compute_taps(5e6, fs, l_min, 64)
            config = engine.get_full_config()
            config.update(sample_rate=fs, waveform_length=len(signal), noise_power_dBm=-200)
            output = SionnaRTAugmentation(config, taps, multi_channel=True).apply(signal, fs)
            self.assertEqual(output.shape, (2, 192))
            for ant in range(2):
                impulse_response = np.pad(taps[0, ant, 0, 0, 0], (l_min, 0))
                expected = np.convolve(signal * np.sqrt(len(signal)), impulse_response)
                np.testing.assert_allclose(output[ant], expected, atol=1e-9)
                self.assertAlmostEqual(np.argmax(np.abs(output[ant])) / fs, 1e-6, delta=0.5 / fs)


if __name__ == "__main__":
    if sys.argv[1:] == ["--real-sionna"]:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        RTTapGridTests.setUpClass()
        RTTapGridTests().check_real_sionna()
    else:
        unittest.main()
