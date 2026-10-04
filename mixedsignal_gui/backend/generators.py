import numpy as np
from typing import Optional


class MATLABWaveformGenerator:

    def __init__(self, matlab_engine: Optional[object], seed=None):
        self.matlab_engine = matlab_engine
        self.seed = seed
        self.last_metadata = {"generator": "matlab"}

    def _matlab_to_numpy(self, data):
        if getattr(data, 'is_complex', False):
            real_part = np.array(data.real, dtype=np.float64).flatten()
            imag_part = np.array(data.imag, dtype=np.float64).flatten()
            return (real_part + 1j * imag_part).astype(np.complex128)

        return np.asarray(data).flatten()

    def generate(self, cfg) -> np.ndarray:
        if self.matlab_engine is None or not getattr(self.matlab_engine, 'is_available', lambda: False)():
            raise RuntimeError("MATLAB engine is not available. Install/configure MATLAB and the MATLAB Engine for Python, or start the engine before generating waveforms.")

        from mixedsignal_gui.backend.core import WaveformConfig

        # waveform_generator.m rounds before it does anything else — line 58 is
        # sps = round(fs*Tsymb) — so MATLAB never errors on a fractional rate,
        # it silently generates at the rounded symbol rate instead.  That is
        # worth refusing, but only where it actually changes the output: the
        # pulse shaper at line 143 is reached by PAM/QAM/PSK alone.  WiFi, LTE
        # and 5G_NR return at lines 119-135, before sps is used at all, and the
        # FSK/FHSS/radar sub-functions round it for themselves.  Refusing those
        # blocked the LTE and 5G_NR presets, whose 3.84 and 30.72 MHz rates are
        # correct for their standards, over a limit that does not apply to them.
        if (getattr(cfg, "is_fractional_sps", False)
                and cfg.modulation in PULSE_SHAPED_MODULATIONS):
            raise RuntimeError(
                f"fs x Tsymb = {cfg.sps_exact:g} samples per symbol is not a whole "
                f"number. waveform_generator.m would round it to {cfg.sps} before "
                f"pulse shaping, generating at a different symbol rate than the one "
                f"requested. Adjust fs or Tsymb, or use the built-in Python "
                f"generator, which realises fractional rates exactly by rational "
                f"resampling.")

        eng = self.matlab_engine.eng
        if self.seed is not None:
            eng.rng(float(self.seed), "twister", nargout=0)
        self.last_metadata = {"generator": "matlab"}

        # Modulations that return symbols as a second output
        symbol_mods = {"PAM", "QAM", "PSK"}
        has_symbols = cfg.modulation in symbol_mods

        if has_symbols:
            data, symbols = eng.waveform_generator(
                float(cfg.output_len),
                float(cfg.fs),
                float(cfg.Tsymb),
                float(cfg.fc),
                float(cfg.M),
                cfg.modulation,
                "alpha", float(cfg.alpha),
                "span", float(cfg.span),
                "pulse_shape", cfg.pulse_shape,
                "output_type", cfg.output_type,
                nargout=2,
            )
            self.last_metadata["baseband_symbols"] = self._matlab_to_numpy(symbols)
        else:
            data = eng.waveform_generator(
                float(cfg.output_len),
                float(cfg.fs),
                float(cfg.Tsymb),
                float(cfg.fc),
                float(cfg.M),
                cfg.modulation,
                "alpha", float(cfg.alpha),
                "span", float(cfg.span),
                "pulse_shape", cfg.pulse_shape,
                "output_type", cfg.output_type,
                nargout=1,
            )

        return self._matlab_to_numpy(data)


# ═══════════════════════════════════════════════════════════════════
# Pure-Python generator
# ═══════════════════════════════════════════════════════════════════

#: Standard Barker sequences, keyed by length.  Used by the phase-coded
#: radar pulse; these are the only lengths for which Barker codes exist.
_BARKER_CODES = {
    2:  [+1, -1],
    3:  [+1, +1, -1],
    4:  [+1, +1, -1, +1],
    5:  [+1, +1, +1, -1, +1],
    7:  [+1, +1, +1, -1, -1, +1, -1],
    11: [+1, +1, +1, -1, -1, -1, +1, -1, -1, +1, -1],
    13: [+1, +1, +1, +1, +1, -1, -1, +1, +1, -1, +1, -1, +1],
}

#: Modulations that go through the pulse shaper — symbols upsampled by
#: samples-per-symbol and filtered.  These are the only ones whose output
#: depends on sps being exact: in waveform_generator.m every other family
#: either returns before line 143 (WiFi/LTE/5G_NR) or rounds sps inside its
#: own sub-function (FSK/FHSS/LFM/Barker/FMCW).
PULSE_SHAPED_MODULATIONS = {"PAM", "QAM", "PSK"}

#: Waveforms that are real communication standards rather than closed-form
#: maths.  MATLAB builds genuine protocol frames for these via its WLAN /
#: LTE / 5G toolboxes; a hand-rolled lookalike would train a classifier that
#: scores well on our own synthetic data and fails on real captures, so they
#: are refused outright rather than approximated.
MATLAB_ONLY_MODULATIONS = {
    "WiFi":   "WLAN Toolbox (802.11ax HE-SU)",
    "LTE":    "LTE Toolbox (downlink RMC R.9, 20 MHz)",
    "5G_NR":  "5G Toolbox (NR downlink, 20 MHz FR1)",
    "Zigbee": "Communications Toolbox (IEEE 802.15.4 O-QPSK)",
}


def unavailable_modulations(matlab_engine) -> dict:
    """Modulations that cannot be generated right now, mapped to the reason.

    Empty when MATLAB is running.  Kept here rather than in the tabs so the
    "what can this engine actually produce" rule lives next to the generators
    that enforce it, and both the Waveform and Evaluate Model tabs agree.
    """
    if matlab_engine is not None and getattr(matlab_engine, "is_available", lambda: False)():
        return {}
    return dict(MATLAB_ONLY_MODULATIONS)


def rcosdesign(alpha: float, span: int, sps: int) -> np.ndarray:
    """Root-raised-cosine taps, equivalent to MATLAB ``rcosdesign(...,'sqrt')``.

    Returns ``span*sps + 1`` taps normalised to unit energy.  The closed form
    is singular at ``t = 0`` and ``t = +/- Ts/(4*alpha)``; both limits are
    substituted analytically rather than nudged, so the taps stay exact.
    """
    n = np.arange(-span * sps / 2.0, span * sps / 2.0 + 1)
    t = n / float(sps)                      # time in symbol periods
    h = np.zeros_like(t, dtype=np.float64)

    if alpha == 0:
        # Degenerates to a sinc; still normalised to unit energy so that
        # signal power does not jump when alpha is dialled down to zero.
        h = np.sinc(t)
        return h / np.sqrt(np.sum(h ** 2))

    # Singular points
    at_zero = np.isclose(t, 0.0)
    at_quarter = np.isclose(np.abs(t), 1.0 / (4.0 * alpha))
    regular = ~(at_zero | at_quarter)

    h[at_zero] = 1.0 - alpha + 4.0 * alpha / np.pi

    if np.any(at_quarter):
        h[at_quarter] = (alpha / np.sqrt(2.0)) * (
            (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha))
            + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha))
        )

    tr = t[regular]
    numerator = (
        np.sin(np.pi * tr * (1.0 - alpha))
        + 4.0 * alpha * tr * np.cos(np.pi * tr * (1.0 + alpha))
    )
    denominator = np.pi * tr * (1.0 - (4.0 * alpha * tr) ** 2)
    h[regular] = numerator / denominator

    return h / np.sqrt(np.sum(h ** 2))


class PythonWaveformGenerator:
    """NumPy/SciPy implementation of ``waveform_functions/waveform_generator.m``.

    Used automatically when the MATLAB engine is unavailable (see
    ``Waveform._ensure_generator``).  MATLAB remains the reference
    implementation; this covers the eight waveforms that are defined by
    closed-form maths.  WiFi / LTE / 5G_NR are refused — see ``MATLAB_ONLY_MODULATIONS``.

    Structure deliberately mirrors the ``.m`` file (samples-per-symbol, filter
    delay, symbol count, pulse shaping, trim, optional upconversion) so the two
    can be compared side by side.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.last_metadata = {"generator": "python"}

    # -- helpers ---------------------------------------------------------

    def _symbols(self, num_symbols: int, M: int, modulation: str) -> np.ndarray:
        """Random unit-average-power symbols.

        Gray coding is intentionally not reproduced: the data is uniform over
        0..M-1, so any bijection onto the constellation gives an identically
        distributed signal.  Only the constellation geometry and the power
        normalisation affect the waveform.
        """
        data = self.rng.integers(0, M, size=num_symbols)

        if modulation == "PAM":
            symbols = (2.0 * data - (M - 1)).astype(np.float64)
            return symbols / np.sqrt(np.mean(np.abs(symbols) ** 2))

        if modulation == "QAM":
            # Square when M is an even power of two (4, 16, 64...), otherwise
            # rectangular with twice as many I levels as Q (8, 32, 128...).
            # WaveformConfig accepts any power of two >= 4, so refusing the
            # non-square cases here would reject input the validator allows.
            #
            # Caveat: MATLAB's qammod is rectangular for M=8 (so we agree) but
            # uses a *cross* constellation for M=32/128, where this differs.
            # The square cases — the defaults everywhere in the app — match.
            bits = int(round(np.log2(M)))
            i_bits = (bits + 1) // 2
            q_bits = bits - i_bits
            n_i, n_q = 1 << i_bits, 1 << q_bits
            i_levels = 2 * np.arange(n_i) - (n_i - 1)
            q_levels = 2 * np.arange(n_q) - (n_q - 1)
            symbols = i_levels[data % n_i] + 1j * q_levels[data // n_i]

            # Normalise by the exact constellation power, matching qammod's
            # 'UnitAveragePower'.  Dividing by the *sample* mean instead (as
            # the .m does for PAM) would make the scale depend on the random
            # draw — around 0.1% jitter run to run, and not what MATLAB does.
            mean_power = np.mean(i_levels ** 2) + np.mean(q_levels ** 2)
            return symbols / np.sqrt(mean_power)

        if modulation == "PSK":
            return np.exp(2j * np.pi * data / M)

        raise ValueError(f"Unsupported modulation type: {modulation}")

    @staticmethod
    def _upconvert(sig_bb: np.ndarray, fs: float, fc: float, output_len: int) -> np.ndarray:
        """Quadrature upconversion: s(t) = I*cos(wc t) - Q*sin(wc t)."""
        sig_bb = sig_bb[:output_len]
        t = np.arange(output_len) / fs
        return (np.real(sig_bb) * np.cos(2 * np.pi * fc * t)
                - np.imag(sig_bb) * np.sin(2 * np.pi * fc * t))

    def _hopping_signal(self, output_len, fs, sps, freq_offsets) -> np.ndarray:
        """Continuous-phase complex baseband from per-interval frequency offsets.

        Phase carries across interval boundaries, which is what makes FSK/FHSS
        continuous-phase rather than a sequence of restarted tones.

        ``sps`` may be fractional.  Boundaries are placed at ``round(k * sps)``
        rather than stepped by a constant integer, so the symbol rate stays
        exact and the intervals absorb the remainder by alternating length
        (3.84 gives 4, 4, 3, 4, 4 ...).  Stepping by a rounded sps instead
        would drift the symbol rate by up to 4% at the rates the LTE and 5G_NR
        presets use, while output_len is computed exactly.
        """
        sig = np.zeros(output_len, dtype=np.complex128)
        phase = 0.0
        for k, f_offset in enumerate(freq_offsets):
            idx = int(round(k * sps))
            end = min(int(round((k + 1) * sps)), output_len)
            n = end - idx
            if n <= 0:
                if idx >= output_len:
                    break
                continue
            step = 2 * np.pi * f_offset / fs
            sig[idx:end] = np.exp(1j * (phase + step * np.arange(n)))
            phase = (phase + step * n) % (2 * np.pi)
        return sig

    # -- waveform families ----------------------------------------------

    def _fsk(self, cfg, sps, output_len) -> np.ndarray:
        freq_sep = 1.0 / cfg.Tsymb
        data = self.rng.integers(0, cfg.M, size=int(np.ceil(output_len / sps)))
        return self._hopping_signal(
            output_len, cfg.fs, sps, (data - (cfg.M - 1) / 2.0) * freq_sep)

    def _fhss(self, cfg, sps, output_len) -> np.ndarray:
        channel_spacing = cfg.fs / (2.0 * cfg.M)
        hops = self.rng.integers(0, cfg.M, size=int(np.ceil(output_len / sps)))
        return self._hopping_signal(
            output_len, cfg.fs, sps, (hops - (cfg.M - 1) / 2.0) * channel_spacing)

    def _lfm(self, cfg, output_len) -> np.ndarray:
        """Pulsed linear-FM chirp, matching the .m's sweep/PRI derivation."""
        fs = cfg.fs
        sweep_bw = cfg.M * fs / 8.0
        pulse_width = min(cfg.Tsymb * 20.0, (output_len - 1) / fs)
        pulse_samples = max(int(round(pulse_width * fs)), 1)
        pri_samples = max(int(round(pulse_samples * 1.5)), pulse_samples + 1)

        t = np.arange(pulse_samples) / fs
        pulse = np.exp(1j * np.pi * (sweep_bw / (pulse_samples / fs)) * t ** 2)

        sig = np.zeros(output_len, dtype=np.complex128)
        for start in range(0, output_len, pri_samples):
            n = min(pulse_samples, output_len - start)
            sig[start:start + n] = pulse[:n]
        return sig

    def _barker(self, cfg, output_len) -> np.ndarray:
        """Phase-coded pulse; M snaps to the nearest valid Barker length."""
        fs = cfg.fs
        lengths = np.array(sorted(_BARKER_CODES))
        num_chips = int(lengths[np.argmin(np.abs(lengths - cfg.M))])
        code = np.array(_BARKER_CODES[num_chips], dtype=np.float64)

        chip_width = min(cfg.Tsymb, (output_len - 1) / (fs * num_chips))
        chip_samples = max(int(round(chip_width * fs)), 1)
        pulse = np.repeat(code, chip_samples).astype(np.complex128)

        pri_samples = max(int(round(len(pulse) * 1.5)), len(pulse) + 1)
        sig = np.zeros(output_len, dtype=np.complex128)
        for start in range(0, output_len, pri_samples):
            n = min(len(pulse), output_len - start)
            sig[start:start + n] = pulse[:n]
        return sig

    def _fmcw(self, cfg, output_len) -> np.ndarray:
        """Continuous triangle-swept FM."""
        fs = cfg.fs
        sweep_bw = cfg.M * fs / 8.0
        sweep_time = min(cfg.Tsymb * 50.0, (output_len - 1) / fs)
        sweep_samples = max(int(round(sweep_time * fs)), 1)

        # Triangle: sweep up over one period, back down over the next.
        up = np.linspace(-sweep_bw / 2, sweep_bw / 2, sweep_samples, endpoint=False)
        freq = np.concatenate([up, up[::-1]])
        reps = int(np.ceil(output_len / len(freq)))
        freq = np.tile(freq, reps)[:output_len]

        phase = 2 * np.pi * np.cumsum(freq) / fs
        return np.exp(1j * phase)

    # -- entry point -----------------------------------------------------

    def generate(self, cfg) -> np.ndarray:
        modulation = cfg.modulation
        if modulation in MATLAB_ONLY_MODULATIONS:
            raise RuntimeError(
                f"{modulation} waveforms require MATLAB and the "
                f"{MATLAB_ONLY_MODULATIONS[modulation]}. The Python generator covers "
                "PAM, QAM, PSK, FSK, FHSS, LFM, Barker and FMCW; it cannot "
                f"synthesise standards-compliant {modulation} frames."
            )

        self.last_metadata = {"generator": "python"}

        # Exact, not cfg.sps: output_len is computed from the exact rate, so
        # timing the symbols by a rounded one would drift them against it.
        sps = cfg.sps_exact
        output_len = cfg.output_len
        baseband = cfg.output_type.lower() == "baseband"

        # Frequency/phase waveforms bypass pulse shaping and produce no symbols
        if modulation in ("FSK", "FHSS", "LFM", "Barker", "FMCW"):
            builder = {
                "FSK": lambda: self._fsk(cfg, sps, output_len),
                "FHSS": lambda: self._fhss(cfg, sps, output_len),
                "LFM": lambda: self._lfm(cfg, output_len),
                "Barker": lambda: self._barker(cfg, output_len),
                "FMCW": lambda: self._fmcw(cfg, output_len),
            }[modulation]
            sig_bb = builder()
            if baseband:
                return sig_bb.astype(np.complex128)
            # FSK/FHSS upconvert by adding the carrier to the running phase in
            # the .m; quadrature upconversion of the same baseband is equivalent.
            return self._upconvert(sig_bb, cfg.fs, cfg.fc, output_len)

        # -- linear modulations: symbols -> pulse shaping -> trim ---------
        from scipy.signal import upfirdn

        # Upsample by p and decimate by q so a fractional samples-per-symbol
        # (LTE's 3.84, say) is realised exactly.  For a whole number q is 1 and
        # p is sps, which reduces to the plain integer path.
        p, q = cfg.sps_ratio

        if cfg.pulse_shape == "rrc":
            # Design at the upsampled rate p, then decimation by q lands the
            # taps at the true symbol spacing.
            h = rcosdesign(cfg.alpha, cfg.span, p)
            filter_delay = int(round(cfg.span * p / 2 / q))
        else:
            h = np.ones(p) / np.sqrt(p)
            filter_delay = 0

        sps_exact = cfg.sps_exact
        num_symbols = int(np.ceil((output_len + 2 * filter_delay) / sps_exact))
        symbols = self._symbols(num_symbols, cfg.M, modulation)
        self.last_metadata["baseband_symbols"] = symbols

        # NOTE argument order: SciPy is upfirdn(h, x, up, down); MATLAB is
        # upfirdn(x, h, p, q).  Swapping these fails silently.
        sig_bb = upfirdn(h, symbols, up=p, down=q)

        # Trim off the filter's group delay, zero-padding if the filtered
        # signal is short (mirrors `sig_bb(end_idx) = 0` in the .m).
        end_idx = filter_delay + output_len
        if end_idx > len(sig_bb):
            sig_bb = np.pad(sig_bb, (0, end_idx - len(sig_bb)))
        sig_bb = sig_bb[filter_delay:end_idx]

        if baseband:
            # Complex even for PAM, which is real-valued: downstream code
            # branches on dtype to decide baseband vs passband.
            return sig_bb.astype(np.complex128)
        return self._upconvert(sig_bb, cfg.fs, cfg.fc, output_len)
