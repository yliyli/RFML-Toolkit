from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class WaveformConfig:
    modulation: str
    fs: float
    Tsymb: float
    fc: float
    M: int
    Nsymb: int

    var: Optional[float] = None
    freq_sep: Optional[float] = None

    alpha: float = 0.35
    span: int = 8
    pulse_shape: str = "rrc"
    # Complex IQ by default: the channel models require it (both the stochastic
    # TDL and Sionna RT paths refuse a real passband array) and the classifiers
    # consume it as two input channels.
    output_type: str = "baseband"

    def __post_init__(self):
        if isinstance(self.M, float) and self.M.is_integer():
            self.M = int(self.M)

        if not (0 <= self.alpha <= 1):
            raise ValueError("alpha must be in [0, 1]")

        if self.pulse_shape not in {"rrc", "rect"}:
            raise ValueError("pulse_shape must be 'rrc' or 'rect'")

        if self.span < 1:
            raise ValueError("span must be >= 1")

        self._validate()

    # Modulations that bypass pulse shaping and don't return symbols
    _NO_SYMBOL_MODS = {"FSK", "FHSS", "LFM", "Barker", "FMCW", "WiFi", "LTE", "5G_NR",
                       "Zigbee", "LoRa"}

    def _validate(self):
        sps = self.fs * self.Tsymb
        # A non-integer samples-per-symbol used to be rejected outright, which
        # ruled out perfectly ordinary sample rates: LTE's 3.84 MHz at a 1 us
        # symbol gives 3.84.  The restriction came from the pulse shaper, not
        # from the signal — upfirdn needs an integer upsampling factor — and
        # the Python generator now handles it by rational resampling instead.
        # MATLABWaveformGenerator still refuses, since MATLAB's upfirdn
        # requires integer factors.
        #
        # One sample per symbol is still the floor, though: below that a symbol
        # has no sample to live on.  The spin boxes reach fs=0.1 MHz with
        # Tsymb=0.01 us (sps=0.001), which produced a 0- or 1-sample waveform
        # with no error at all, or a bare "Invalid number of FFT data points".
        # The old integer check happened to exclude these; requiring >= 1 is
        # the constraint that was actually doing the work.
        if sps < 1:
            raise ValueError(
                f"fs x Tsymb = {sps:g} gives less than one sample per symbol. "
                f"Increase the sample rate or the symbol period so that "
                f"fs x Tsymb >= 1.")

        if self.output_type == "passband" and self.fc >= self.fs / 2:
            raise ValueError("fc must be < fs/2")

        if self.modulation == "FSK":
            self._require_power_of_2()
        elif self.modulation == "QAM":
            self._require_power_of_2(min_val=4)
        elif self.modulation == "PAM":
            if self.M < 2:
                raise ValueError("PAM requires M >= 2")
        elif self.modulation == "PSK":
            self._require_power_of_2()
        elif self.modulation == "FHSS":
            if self.M < 2:
                raise ValueError("FHSS requires M >= 2")
        elif self.modulation in ("LFM", "FMCW"):
            if self.M < 2:
                raise ValueError(f"{self.modulation} requires M >= 2")
        elif self.modulation == "Barker":
            # M maps to nearest valid Barker length (2,3,4,5,7,11,13)
            if self.M < 2:
                raise ValueError("Barker requires M >= 2")
        elif self.modulation in ("WiFi", "LTE", "5G_NR", "Zigbee", "LoRa"):
            # Standards waveforms — M is unused by the MATLAB generator
            pass
        else:
            raise ValueError(f"Unknown modulation: {self.modulation}")

    def _require_power_of_2(self, min_val=2):
        if self.M < min_val or (self.M & (self.M - 1)) != 0:
            raise ValueError(f"{self.modulation} requires M to be power of 2 >= {min_val}")

    @property
    def sps(self) -> int:
        # round(), not int(), to match `sps = round(fs * Tsymb)` in
        # waveform_generator.m.  _validate() only requires the product to be
        # within 1e-9 of an integer, and floating point routinely lands just
        # below one: the UI's fs=0.10 MHz, Tsymb=10.00 us gives
        # 0.9999999999999999, which int() truncates to 0 — an empty waveform,
        # and a zero output_len handed to MATLAB.
        return int(round(self.fs * self.Tsymb))

    @property
    def sps_exact(self) -> float:
        """Samples per symbol without rounding, e.g. 3.84 for LTE at 1 us."""
        return float(self.fs) * float(self.Tsymb)

    #: Largest upsampling factor the pulse shaper will accept.  The RRC is
    #: designed at ``p`` samples per symbol, so the filter is ``span*p + 1``
    #: taps and upfirdn's intermediate signal is ``p`` times the symbol count.
    _MAX_UPSAMPLE = 50_000

    @property
    def sps_ratio(self) -> tuple:
        """``sps_exact`` as a fraction ``(p, q)``.

        Pulse shaping upsamples by ``p`` and decimates by ``q``, so a
        fractional samples-per-symbol is realised without rounding.

        The denominator has to be bounded, because ``p`` sets both the filter
        length and the intermediate rate.  A bound of 1000 covers every rate
        the app ships exactly (3.84 -> 96/25, 30.72 -> 768/25), but it is not
        exact for every rate the spin boxes can reach: they step in hundredths
        of a MHz and of a microsecond, so ``sps`` is a multiple of 1e-4 and its
        exact denominator can be up to 10000.  Escalating catches those —
        1.0005 becomes 2001/2000 rather than collapsing to 1, which was a 0.05%
        symbol-rate error, the worst case over the whole spin-box grid.

        Escalation stops if ``p`` would exceed ``_MAX_UPSAMPLE``, so a
        pathological rate degrades to a close approximation instead of building
        a million-tap filter.  Over the full grid of reachable (fs, Tsymb)
        pairs the residual error is then at most ~1e-9 relative.
        """
        from fractions import Fraction
        v = self.sps_exact
        if v > self._MAX_UPSAMPLE:
            raise ValueError(
                f"fs x Tsymb = {v:g} requires more than "
                f"{self._MAX_UPSAMPLE:,} samples per symbol.")

        desired_denominator = 1_000
        coarse = Fraction(v).limit_denominator(desired_denominator)
        if abs(float(coarse) - v) > 1e-12 * abs(v):
            desired_denominator = 10_000

        frac = Fraction(v).limit_denominator(desired_denominator)
        if frac.numerator > self._MAX_UPSAMPLE:
            # Numerator grows approximately as v * denominator. Select the
            # finest denominator that cannot exceed the resource cap.
            bounded_denominator = max(1, int(self._MAX_UPSAMPLE // v))
            frac = Fraction(v).limit_denominator(bounded_denominator)
            while frac.numerator > self._MAX_UPSAMPLE and bounded_denominator > 1:
                bounded_denominator -= 1
                frac = Fraction(v).limit_denominator(bounded_denominator)
        return frac.numerator, frac.denominator

    @property
    def is_fractional_sps(self) -> bool:
        return abs(self.sps_exact - round(self.sps_exact)) > 1e-9

    @property
    def output_len(self) -> int:
        return int(round(self.sps_exact * self.Nsymb))


class Waveform:

    def __init__(
        self,
        *,
        fs: float,
        Tsymb: float,
        Nsymb: int,
        fc: float,
        M: int,
        modulation: str,
        matlab_engine=None,
        generator_impl=None,
        var: Optional[float] = None,
        freq_sep: Optional[float] = None,
        alpha: float = 0.35,
        span: int = 8,
        pulse_shape: str = "rrc",
        output_type: str = "baseband",
        seed: Optional[int] = None,
    ):
        self.config = WaveformConfig(
            modulation=modulation,
            fs=fs,
            Tsymb=Tsymb,
            fc=fc,
            M=M,
            Nsymb=Nsymb,
            var=var,
            freq_sep=freq_sep,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
            output_type=output_type,
        )

        self.matlab_engine = matlab_engine
        self.seed = seed
        self._generator = generator_impl
        self._data: Optional[np.ndarray] = None
        self._metadata = {}

    def _ensure_generator(self):
        """Pick a generator: MATLAB when it is usable, otherwise pure Python.

        MATLAB stays the reference implementation for other families. LoRa
        uses the direct Python chirp construction even with a live engine.
        An explicitly injected ``generator_impl`` always
        takes precedence, which is how callers and tests force one or the other.

        Python also handles a fractional samples-per-symbol on a pulse-shaped
        modulation.  waveform_generator.m rounds sps before pulse shaping, so
        MATLAB would generate at a symbol rate the caller did not ask for; the
        Python generator realises the rate exactly by rational resampling.  It
        therefore wins that case on merit rather than availability, and the
        choice stays visible because ``last_metadata`` records which engine ran.
        """
        if self._generator is not None:
            return

        from mixedsignal_gui.backend.generators import (
            MATLABWaveformGenerator, PythonWaveformGenerator,
            PULSE_SHAPED_MODULATIONS)

        engine = self.matlab_engine
        engine_live = (engine is not None
                       and getattr(engine, "is_available", lambda: False)())
        prefer_python = (self.config.modulation == "LoRa" or (
            self.config.is_fractional_sps and self.config.modulation in PULSE_SHAPED_MODULATIONS))

        if engine_live and not prefer_python:
            self._generator = MATLABWaveformGenerator(engine, seed=self.seed)
        else:
            self._generator = PythonWaveformGenerator(seed=self.seed)

    def generate(self) -> np.ndarray:
        self._ensure_generator()
        self._data = self._generator.generate(self.config)
        self._metadata = dict(getattr(self._generator, "last_metadata", {}) or {})
        return self._data

    def generate_data(self):
        self.generate()
        return self._data

    def get_data(self):
        if self._data is None:
            raise ValueError("Call generate_data() first")
        return self._data

    def get_sps(self):
        return self.config.sps

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("Waveform not generated yet")
        return self._data

    @property
    def metadata(self) -> dict:
        return dict(self._metadata)
