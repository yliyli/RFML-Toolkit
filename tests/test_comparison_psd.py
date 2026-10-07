"""PSD dB display preserves density, relative scaling, and input samples."""
import unittest

import numpy as np
from matplotlib.figure import Figure
from scipy import signal

from mixedsignal_gui.widgets.comparison_widget import ComparisonWidget


class ComparisonPSDTests(unittest.TestCase):
    def test_db_density_for_real_and_complex_and_no_mutation(self):
        for complex_iq in (False, True):
            with self.subTest(complex_iq=complex_iq):
                rng = np.random.default_rng(42)
                clean = rng.normal(size=2048)
                if complex_iq:
                    clean = clean + 1j * rng.normal(size=2048)
                augmented = clean * .01
                original = clean.copy(), augmented.copy()
                ax = Figure().subplots()
                ComparisonWidget._plot_psd(None, ax, clean, augmented, 30.72e6, complex_iq)
                f, density = signal.welch(clean, 30.72e6, nperseg=1024,
                                          return_onesided=not complex_iq)
                if complex_iq:
                    f, density = np.fft.fftshift(f), np.fft.fftshift(density)
                np.testing.assert_allclose(ax.lines[0].get_xdata(), f / 1e6)
                np.testing.assert_allclose(ax.lines[0].get_ydata(), 10 * np.log10(density))
                np.testing.assert_allclose(ax.lines[1].get_ydata(),
                                           ax.lines[0].get_ydata() - 40)
                np.testing.assert_array_equal(clean, original[0])
                np.testing.assert_array_equal(augmented, original[1])
                self.assertEqual(ax.get_yscale(), 'linear')
                self.assertIn('Uncalibrated', ax.get_ylabel())
                self.assertNotIn('dBm', ax.get_ylabel())

    def test_zero_signals_have_finite_display(self):
        ax = Figure().subplots()
        zeros = np.zeros(2048)
        ComparisonWidget._plot_psd(None, ax, zeros, zeros, 1e6, False)
        for line in ax.lines:
            self.assertTrue(np.isfinite(line.get_ydata()).all())
