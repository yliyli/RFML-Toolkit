"""
Comparison visualization widget for clean vs augmented signals.

Uses the same DSP routines as the Waveform Generation tab
(via signal_utils) so that constellations, IQ trajectories,
and spectrograms are visually consistent.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QApplication, QTabWidget, QFrame
from PySide6.QtCore import QEvent
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from mixedsignal_gui.widgets.signal_utils import demodulate_to_symbols, extract_fsk_iq, extract_fhss_iq


def _hilbert_iq(x, fs, fc):
    """Extract I/Q via Hilbert-transform downconversion.

    Works for any real passband signal regardless of modulation:
    multiply by exp(-j*2*pi*fc*t) to shift to baseband, then
    low-pass by keeping only the analytic (positive-freq) part.
    Returns decimated I and Q arrays suitable for scatter plots.
    """
    x = np.asarray(x).ravel().astype(np.float64)
    t = np.arange(len(x)) / fs
    # Downconvert
    analytic = signal.hilbert(x)
    bb = analytic * np.exp(-1j * 2 * np.pi * fc * t)
    # Decimate for plotting (every 4th sample keeps plot fast)
    step = max(1, len(bb) // 4096)
    bb = bb[::step]
    return bb.real, bb.imag


def _ideal_constellation_points(modulation, M):
    """Return ideal constellation points for common symbol modulations."""
    M = int(M) if M is not None else 2

    if modulation == 'PAM':
        levels = np.arange(-(M - 1), M, 2, dtype=float)
        scale = np.sqrt(np.mean(levels ** 2)) if len(levels) else 1.0
        return levels / max(scale, 1e-12)

    if modulation == 'PSK':
        phases = 2 * np.pi * np.arange(M) / M
        return np.exp(1j * phases)

    if modulation == 'QAM':
        side = int(np.sqrt(M))
        if side * side != M:
            side = max(2, side)
        levels = np.arange(-(side - 1), side, 2, dtype=float)
        grid = np.array([x + 1j * y for y in levels for x in levels], dtype=np.complex128)
        scale = np.sqrt(np.mean(np.abs(grid) ** 2)) if len(grid) else 1.0
        return grid / max(scale, 1e-12)

    return np.array([], dtype=np.complex128)


def _snap_to_constellation(symbols, modulation, M):
    """Snap recovered symbols to the nearest ideal constellation points."""
    ideal = _ideal_constellation_points(modulation, M)
    if len(symbols) == 0 or len(ideal) == 0:
        return np.asarray(symbols)

    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    distances = np.abs(symbols[:, None] - ideal[None, :])
    nearest = np.argmin(distances, axis=1)
    return ideal[nearest]


# Modulation-aware spectrogram presets
_SPECTROGRAM_PRESETS = {
    "PAM":    {"nperseg": 512,  "overlap": 0.70, "window": "hann",     "vmin_pct": 10, "vmax_pct": 95},
    "QAM":    {"nperseg": 1024, "overlap": 0.75, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 95},
    "PSK":    {"nperseg": 1024, "overlap": 0.75, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 95},
    "ASK":    {"nperseg": 512,  "overlap": 0.70, "window": "hann",     "vmin_pct": 10, "vmax_pct": 95},
    "FSK":    {"nperseg": 2048, "overlap": 0.85, "window": "blackman", "vmin_pct": 3,  "vmax_pct": 97},
    "OFDM":   {"nperseg": 2048, "overlap": 0.80, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 90},
    "FHSS":   {"nperseg": 2048, "overlap": 0.85, "window": "blackman", "vmin_pct": 3,  "vmax_pct": 97},
    # Radar waveforms — wideband chirps benefit from large windows + high overlap
    "LFM":    {"nperseg": 2048, "overlap": 0.85, "window": "hann",     "vmin_pct": 3,  "vmax_pct": 97},
    "Barker": {"nperseg": 1024, "overlap": 0.80, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 95},
    "FMCW":   {"nperseg": 2048, "overlap": 0.85, "window": "hann",     "vmin_pct": 3,  "vmax_pct": 97},
    # Standards waveforms — OFDM-based, use matching presets
    "WiFi":   {"nperseg": 2048, "overlap": 0.80, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 90},
    "LTE":    {"nperseg": 2048, "overlap": 0.80, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 90},
    "5G_NR":  {"nperseg": 2048, "overlap": 0.80, "window": "hann",     "vmin_pct": 5,  "vmax_pct": 90},
}
_DEFAULT_PRESET = {"nperseg": 1024, "overlap": 0.75, "window": "hann", "vmin_pct": 5, "vmax_pct": 95}


def _mpl_theme():
    p = QApplication.palette()
    return (p.window().color().name(),
            p.windowText().color().name(),
            p.base().color().name())


def _apply_mpl_theme(fig, *axes):
    bg, fg, axes_bg = _mpl_theme()
    fig.patch.set_facecolor(bg)
    for ax in axes:
        ax.set_facecolor(axes_bg)
        ax.tick_params(colors=fg)
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.title.set_color(fg)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg)
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(axes_bg)
            for text in leg.get_texts():
                text.set_color(fg)
    
    # Handle colorbars
    for ax in fig.get_axes():
        if hasattr(ax, '_colorbar'):
            cbar = ax._colorbar
            cbar.ax.tick_params(colors=fg)
            if cbar.ax.yaxis.label:
                cbar.ax.yaxis.label.set_color(fg)
            for spine in cbar.ax.spines.values():
                spine.set_edgecolor(fg)


def _theme_toolbar(toolbar):
    """Style NavigationToolbar2QT to match the current Qt palette."""
    p = QApplication.palette()
    bg   = p.window().color().name()
    fg   = p.windowText().color().name()
    btn  = p.button().color().name()
    toolbar.setStyleSheet(f"""
        QToolBar {{
            background: {bg};
            border: none;
            spacing: 2px;
        }}
        QToolButton {{
            background: {btn};
            color: {fg};
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 2px;
        }}
        QToolButton:hover  {{ border-color: {fg}; }}
        QToolButton:checked {{ background: {fg}; color: {bg}; }}
    """)


class ComparisonWidget(QWidget):
    """Widget to display clean vs augmented signal comparison with overlays"""

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        # Antenna selector (hidden until multi-channel data arrives)
        self._ant_bar = QHBoxLayout()
        self._ant_label = QLabel("Antenna:")
        self._ant_combo = QComboBox()
        self._ant_combo.setFixedWidth(140)
        self._ant_combo.currentIndexChanged.connect(self._on_antenna_changed)
        self._ant_bar.addWidget(self._ant_label)
        self._ant_bar.addWidget(self._ant_combo)
        self._ant_bar.addStretch()
        self._ant_label.setVisible(False)
        self._ant_combo.setVisible(False)
        layout.addLayout(self._ant_bar)

        # Each row gets its own tab so the plot groups are easier to inspect.
        self.plot_tabs = QTabWidget()
        self._tab_specs = {}
        self._create_plot_tab("signals", "Signals", (14, 4.5))
        self._create_plot_tab("spectrograms", "Spectrograms", (14, 4.5))
        self._create_plot_tab("constellation", "Constellation", (8, 6))
        layout.addWidget(self.plot_tabs)
        self.setLayout(layout)

        self._last_plot_args = {}

        self.clear_plots()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.PaletteChange:
            for spec in self._tab_specs.values():
                _theme_toolbar(spec['toolbar'])
                bg, _, _ = _mpl_theme()
                spec['figure'].patch.set_facecolor(bg)
                for ax in spec['figure'].get_axes():
                    _apply_mpl_theme(spec['figure'], ax)
                spec['canvas'].draw_idle()
            self._replot_with_theme()
        super().changeEvent(event)
    
    def _replot_with_theme(self):
        for spec in self._tab_specs.values():
            _theme_toolbar(spec['toolbar'])
            bg, _, _ = _mpl_theme()
            spec['figure'].patch.set_facecolor(bg)
        if self._last_plot_args:
            self.plot_comparison(**self._last_plot_args)

    def _on_antenna_changed(self, _idx):
        """Re-plot with the newly selected antenna."""
        if self._last_plot_args:
            self.plot_comparison(**self._last_plot_args)

    def clear_plots(self):
        """Clear all subplots"""
        for spec in self._tab_specs.values():
            spec['figure'].clear()
            spec['canvas'].draw()

    def _create_plot_tab(self, key, title, fig_size):
        """Create a tab containing its own figure, canvas, and toolbar."""
        bg, _, _ = _mpl_theme()
        figure = Figure(figsize=fig_size, facecolor=bg)
        canvas = FigureCanvas(figure)
        canvas.setStyleSheet("background: transparent;")
        toolbar = NavigationToolbar(canvas, self)
        _theme_toolbar(toolbar)

        container = QFrame()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(toolbar)
        container_layout.addWidget(canvas)

        self._tab_specs[key] = {
            'figure': figure,
            'canvas': canvas,
            'toolbar': toolbar,
            'container': container,
        }
        self.plot_tabs.addTab(container, title)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def plot_comparison(self, clean_signal, augmented_signal, fs,
                        fc=None, sps=None, modulation=None,
                        M=None, alpha=0.35, span=8,
                        pulse_shape='rrc', nsymb=None):
        self._last_plot_args = {
            'clean_signal': clean_signal,
            'augmented_signal': augmented_signal,
            'fs': fs,
            'fc': fc,
            'sps': sps,
            'modulation': modulation,
            'M': M,
            'alpha': alpha,
            'span': span,
            'pulse_shape': pulse_shape,
            'nsymb': nsymb
        }

        for spec in self._tab_specs.values():
            spec['figure'].clear()

        # Multi-channel augmented signal: select antenna via combo box
        is_mc = isinstance(augmented_signal, np.ndarray) and augmented_signal.ndim == 2
        if is_mc:
            num_ant = augmented_signal.shape[0]
            # Populate combo box (block signals to avoid re-entrant replot)
            self._ant_combo.blockSignals(True)
            prev_idx = self._ant_combo.currentIndex()
            self._ant_combo.clear()
            for i in range(num_ant):
                self._ant_combo.addItem(f"Antenna {i}")
            # Restore previous selection if still valid
            if 0 <= prev_idx < num_ant:
                self._ant_combo.setCurrentIndex(prev_idx)
            else:
                self._ant_combo.setCurrentIndex(0)
            self._ant_combo.blockSignals(False)
            self._ant_label.setVisible(True)
            self._ant_combo.setVisible(True)

            ant_idx = self._ant_combo.currentIndex()
            aug_for_plot = augmented_signal[ant_idx]
            mc_label = f" (Ant {ant_idx} of {num_ant})"
        else:
            self._ant_label.setVisible(False)
            self._ant_combo.setVisible(False)
            aug_for_plot = augmented_signal
            mc_label = ""

        any_complex = np.iscomplexobj(clean_signal) or np.iscomplexobj(aug_for_plot)

        # Tab 1: time domain + spectrum
        sig_fig = self._tab_specs['signals']['figure']
        ax1 = sig_fig.add_subplot(1, 2, 1)
        ax2 = sig_fig.add_subplot(1, 2, 2)
        self._plot_time_domain(ax1, clean_signal, aug_for_plot, fs)
        self._plot_psd(ax2, clean_signal, aug_for_plot, fs, any_complex)
        _apply_mpl_theme(sig_fig, ax1, ax2)
        sig_fig.tight_layout()
        self._tab_specs['signals']['canvas'].draw()

        # Tab 2: spectrograms
        spec_fig = self._tab_specs['spectrograms']['figure']
        ax3 = spec_fig.add_subplot(1, 2, 1)
        ax4 = spec_fig.add_subplot(1, 2, 2)
        self._plot_spectrogram(ax3, clean_signal, fs, any_complex, modulation, 'Clean Signal Spectrogram')
        self._plot_spectrogram(ax4, aug_for_plot, fs, any_complex, modulation, f'Augmented Signal Spectrogram{mc_label}')
        _apply_mpl_theme(spec_fig, ax3, ax4)
        spec_fig.tight_layout()
        self._tab_specs['spectrograms']['canvas'].draw()

        # Tab 3: constellation / IQ trajectory
        const_fig = self._tab_specs['constellation']['figure']
        ax5 = const_fig.add_subplot(1, 1, 1)
        self._plot_constellation(ax5, clean_signal, aug_for_plot, fs,
                     fc=fc, sps=sps, modulation=modulation,
                     M=M, alpha=alpha, span=span,
                     pulse_shape=pulse_shape, nsymb=nsymb)
        _apply_mpl_theme(const_fig, ax5)
        const_fig.tight_layout()
        self._tab_specs['constellation']['canvas'].draw()

    # ------------------------------------------------------------------
    # Subplot helpers
    # ------------------------------------------------------------------
    def _plot_time_domain(self, ax, clean, augmented, fs):
        n_samples = min(200, len(clean))
        t = np.arange(n_samples) / fs * 1e6  # µs

        any_complex = np.iscomplexobj(clean) or np.iscomplexobj(augmented)
        if any_complex:
            ax.plot(t, np.real(clean[:n_samples]), 'b-', linewidth=1.5, label='Clean I', alpha=0.8)
            ax.plot(t, np.imag(clean[:n_samples]), 'b--', linewidth=1.0, label='Clean Q', alpha=0.6)
            ax.plot(t, np.real(augmented[:n_samples]), 'r-', linewidth=0.8, label='Aug I', alpha=0.6)
            ax.plot(t, np.imag(augmented[:n_samples]), 'r--', linewidth=0.6, label='Aug Q', alpha=0.4)
        else:
            ax.plot(t, clean[:n_samples], 'b-', linewidth=1.5, label='Clean', alpha=0.8)
            ax.plot(t, augmented[:n_samples], 'r-', linewidth=0.8, label='Augmented', alpha=0.6)

        ax.set_title('Time Domain: Clean vs Augmented')
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Amplitude')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_psd(self, ax, clean, augmented, fs, any_complex):
        nperseg_psd = min(1024, len(clean))
        if any_complex:
            f_psd, psd_clean = signal.welch(clean, fs, nperseg=nperseg_psd, return_onesided=False)
            _, psd_aug = signal.welch(augmented, fs, nperseg=nperseg_psd, return_onesided=False)
            f_psd = np.fft.fftshift(f_psd)
            psd_clean = np.fft.fftshift(psd_clean)
            psd_aug = np.fft.fftshift(psd_aug)
        else:
            f_psd, psd_clean = signal.welch(clean, fs, nperseg=nperseg_psd)
            _, psd_aug = signal.welch(augmented, fs, nperseg=nperseg_psd)

        # Display only: common unit reference, no per-curve normalization.
        # Floor exact zeros to keep the logarithm finite. Samples are unchanged.
        floor = np.finfo(np.float64).tiny
        ax.plot(f_psd / 1e6, 10 * np.log10(np.maximum(psd_clean, floor)),
                'b-', linewidth=2, label='Clean')
        ax.plot(f_psd / 1e6, 10 * np.log10(np.maximum(psd_aug, floor)),
                'r-', linewidth=1, alpha=0.7, label='Augmented')

        ax.set_title('Power Spectrum: Clean vs Augmented')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('PSD (dB re 1 sample-unit²/Hz)\nUncalibrated')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_spectrogram(self, ax, x, fs, is_complex, modulation, title):
        preset = _SPECTROGRAM_PRESETS.get(modulation, _DEFAULT_PRESET)
        nperseg = preset["nperseg"]

        # Cap nperseg so the spectrogram has at least a few time segments.
        # pcolormesh needs ≥2 columns to render; targeting ≥4 for a useful plot.
        max_nperseg = max(16, len(x) // 4)
        if nperseg > max_nperseg:
            nperseg = max_nperseg
        noverlap = int(nperseg * preset["overlap"])
        if noverlap >= nperseg:
            noverlap = nperseg - 1

        if is_complex:
            f, t_s, Sxx = signal.spectrogram(
                x, fs=fs, window=preset["window"],
                nperseg=nperseg, noverlap=noverlap,
                scaling='density', mode='psd',
                return_onesided=False,
            )
            f = np.fft.fftshift(f)
            Sxx = np.fft.fftshift(Sxx, axes=0)
        else:
            f, t_s, Sxx = signal.spectrogram(
                x, fs=fs, window=preset["window"],
                nperseg=nperseg, noverlap=noverlap,
                scaling='density', mode='psd',
            )

        Sxx_dB = 10 * np.log10(Sxx + 1e-10)
        vmin = np.percentile(Sxx_dB, preset["vmin_pct"])
        vmax = np.percentile(Sxx_dB, preset["vmax_pct"])

        im = ax.pcolormesh(
            t_s * 1e6, f / 1e6, Sxx_dB,
            shading='gouraud', cmap='viridis',
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Frequency (MHz)')
        _, fg, _ = _mpl_theme()
        cb = ax.figure.colorbar(im, ax=ax, label='dB/Hz')
        cb.ax.yaxis.set_tick_params(color=fg)
        cb.ax.yaxis.label.set_color(fg)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=fg)

    def _plot_constellation(self, ax, clean, augmented, fs,
                            fc, sps, modulation, M, alpha, span,
                            pulse_shape, nsymb):
        """Plot constellation or IQ trajectory for both clean and augmented."""
        if sps is None or modulation is None:
            ax.text(0.5, 0.5, 'Constellation not available\n(requires sps and modulation)',
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        sps = int(sps)

        # Modulations that use IQ trajectory (no symbol demod)
        _IQ_TRAJECTORY_MODS = {'FSK', 'FHSS', 'LFM', 'Barker', 'FMCW', 'WiFi', 'LTE', '5G_NR'}

        try:
            if modulation in _IQ_TRAJECTORY_MODS:
                # IQ trajectory via analytic signal (Hilbert transform)
                if modulation == 'FSK':
                    if M is None:
                        M = 4
                    I_c, Q_c = extract_fsk_iq(clean, fs, fc, sps, M)
                    I_a, Q_a = extract_fsk_iq(augmented, fs, fc, sps, M)
                    title_str = f'{int(M)}-FSK IQ Trajectory'
                elif modulation == 'FHSS':
                    if M is None:
                        M = 4
                    I_c, Q_c = extract_fhss_iq(clean, fs, fc, sps, M)
                    I_a, Q_a = extract_fhss_iq(augmented, fs, fc, sps, M)
                    title_str = 'FHSS IQ Trajectory'
                else:
                    # Generic IQ via Hilbert downconversion for radar/standards
                    I_c, Q_c = _hilbert_iq(clean, fs, fc)
                    I_a, Q_a = _hilbert_iq(augmented, fs, fc)
                    title_str = f'{modulation} IQ Trajectory'

                ax.scatter(I_c, Q_c, c='blue', s=8, alpha=0.4, label='Clean')
                ax.scatter(I_a, Q_a, c='red', s=8, alpha=0.25, label='Augmented')
                ax.set_title(f'{title_str}: Clean vs Augmented')

            else:
                # PAM / QAM / PSK — matched-filter demodulation
                sym_clean = None
                if modulation in {'PAM', 'QAM', 'PSK'}:
                    baseband_clean = self._last_plot_args.get('baseband_symbols') if hasattr(self, '_last_plot_args') else None
                    if baseband_clean is not None and len(np.asarray(baseband_clean).reshape(-1)) > 0:
                        sym_clean = np.asarray(baseband_clean).reshape(-1)

                if sym_clean is None:
                    sym_clean = demodulate_to_symbols(
                        clean, fs, fc, sps, alpha=alpha, span=span,
                        pulse_shape=pulse_shape, nsymb=nsymb,
                    )

                sym_clean = _snap_to_constellation(sym_clean, modulation, M)
                sym_aug = demodulate_to_symbols(
                    augmented, fs, fc, sps, alpha=alpha, span=span,
                    pulse_shape=pulse_shape, nsymb=nsymb,
                )
                ax.scatter(np.real(sym_clean), np.imag(sym_clean),
                           c='blue', s=15, alpha=0.5, label='Clean')
                ax.scatter(np.real(sym_aug), np.imag(sym_aug),
                           c='red', s=15, alpha=0.3, label='Augmented')
                if M is not None:
                    ax.set_title(f'{int(M)}-{modulation} Constellation: Clean vs Augmented')
                else:
                    ax.set_title(f'{modulation} Constellation: Clean vs Augmented')

        except Exception as e:
            ax.text(0.5, 0.5, f'Constellation error:\n{str(e)}',
                    ha='center', va='center', fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        ax.set_xlabel('In-Phase (I)')
        ax.set_ylabel('Quadrature (Q)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

        # Fixed axis limits derived from the clean constellation extent
        # so the scale stays locked as the user sweeps the SNR slider.
        # 50 % margin gives room for noise to visibly spread outward.
        if modulation in _IQ_TRAJECTORY_MODS:
            all_I = np.concatenate([I_c, I_a])
            all_Q = np.concatenate([Q_c, Q_a])
        else:
            all_I = np.real(sym_clean)
            all_Q = np.imag(sym_clean)
        half = max(np.max(np.abs(all_I)), np.max(np.abs(all_Q)), 0.1) * 1.5
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect('equal')
