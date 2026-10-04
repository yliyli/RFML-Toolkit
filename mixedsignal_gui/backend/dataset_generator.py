from PySide6.QtCore import QThread, Signal
import numpy as np
import os
from .experiment_settings import experiment_seed


class DatasetGeneratorThread(QThread):
    """Generate N waveform samples per modulation class using the MATLAB engine.

    Saves each sample to: <output_dir>/<ModulationType>/data_<index>.npy
    Emits `sample_progress` after each sample and `generation_finished` when done.
    """
    sample_progress = Signal(int, int)      # (current_count, total_count)
    generation_finished = Signal(str)       # output directory path
    error_occurred = Signal(str)            # error message

    def __init__(self, matlab_engine, modulations, samples_per_class, output_dir,
                 waveform_params=None, m_map_override=None):
        """
        Args:
            matlab_engine: Shared MATLAB engine wrapper (backend.matlab_engine.MatlabEngine)
            modulations: list of modulation type strings, e.g. ['PAM', 'QAM', 'PSK']
            samples_per_class: int — number of samples to generate for each modulation
            output_dir: base directory to write class folders into
            waveform_params: dict of default waveform parameters; individual fields may be
                             randomized per sample inside `_randomize_params`.
            m_map_override: dict mapping modulation name to list of M values to randomly
                            choose from per sample. Overrides the built-in _M_MAP entries.
        """
        super().__init__()
        self.matlab_engine = matlab_engine
        self.modulations = list(modulations)
        self.samples_per_class = int(samples_per_class)
        self.output_dir = output_dir
        self.waveform_params = waveform_params or {}
        self.m_map_override = m_map_override or {}
        self._stop = False
        self.run_seed = experiment_seed()
        self.rng = np.random.default_rng(self.run_seed)

    def stop(self):
        self._stop = True

    # Default parameter ranges used when the user does not specify overrides
    _DEFAULTS = {
        'fs':          100e3,
        'Tsymb':       1e-3,
        'Nsymb':       128,
        'fc':          10e3,
        'alpha':       0.35,
        'span':        8,
        'pulse_shape': 'rrc',
    }

    # Per-waveform default overrides.
    # Standards-based waveforms need higher sample rates because their MATLAB
    # generators produce wideband signals (20 MHz+ native) that are resampled
    # to the target fs.  Radar waveforms benefit from larger fc headroom.
    _WAVEFORM_DEFAULTS = {
        'LFM':    {'fs': 1e6, 'fc': 200e3, 'Tsymb': 1e-4},
        'Barker': {'fs': 1e6, 'fc': 200e3, 'Tsymb': 5e-5},
        'FMCW':   {'fs': 1e6, 'fc': 200e3, 'Tsymb': 1e-4},
        'WiFi':   {'fs': 1e6, 'fc': 250e3},
        'LTE':    {'fs': 1e6, 'fc': 250e3},
        '5G_NR':  {'fs': 1e6, 'fc': 250e3},
    }

    # How M maps to each modulation
    _M_MAP = {
        'PAM':   [2, 4, 8],
        'QAM':   [4, 16, 64],
        'PSK':   [2, 4, 8],
        'FSK':   [2, 4, 8],
        'FHSS':  [4, 8, 16],
        'OFDM':  [4, 16, 64],
        # Radar — M controls bandwidth/chips
        'LFM':    [2, 4, 8],
        'Barker': [5, 7, 11, 13],
        'FMCW':   [2, 4, 8],
        # Standards — M is unused but must be valid
        'WiFi':  [4],
        'LTE':   [4],
        '5G_NR': [4],
    }

    # Variance range for additive noise
    _VAR_RANGE = (0.001, 0.05)

    def _randomize_params(self, modulation):
        """Return a dict of waveform parameters with light randomization."""
        rng = self.rng

        params = dict(self._DEFAULTS)

        # Apply per-waveform default overrides BEFORE user overrides
        wf_overrides = self._WAVEFORM_DEFAULTS.get(modulation, {})
        params.update(wf_overrides)

        params.update(self.waveform_params)  # user overrides

        # Pick a random M for the modulation type (user-editable choices)
        m_choices = self.m_map_override.get(modulation, self._M_MAP.get(modulation, [4]))
        params['M'] = int(rng.choice(m_choices))

        # Randomize noise variance
        params['var'] = float(rng.uniform(*self._VAR_RANGE))

        # Small jitter on carrier frequency (±10 %)
        base_fc = params.get('fc', 10e3)
        params['fc'] = float(base_fc * rng.uniform(0.9, 1.1))

        params['modulation'] = modulation

        # FSK specific — default freq_sep matches MATLAB's 1/Tsymb
        if modulation == 'FSK':
            params['freq_sep'] = float(params.get('freq_sep', 1.0 / params['Tsymb']))

        return params

    def run(self):
        from mixedsignal_gui.backend.core import Waveform

        total = self.samples_per_class * len(self.modulations)
        count = 0

        for mod in self.modulations:
            class_dir = os.path.join(self.output_dir, mod)
            os.makedirs(class_dir, exist_ok=True)

            for i in range(self.samples_per_class):
                if self._stop:
                    self.generation_finished.emit(self.output_dir)
                    return

                try:
                    params = self._randomize_params(mod)
                    waveform = Waveform(
                        fs=params['fs'],
                        Tsymb=params['Tsymb'],
                        Nsymb=params['Nsymb'],
                        fc=params['fc'],
                        M=params['M'],
                        modulation=params['modulation'],
                        var=params.get('var'),
                        matlab_engine=self.matlab_engine,
                        alpha=params.get('alpha', 0.35),
                        span=params.get('span', 8),
                        pulse_shape=params.get('pulse_shape', 'rrc'),
                        seed=(self.run_seed + count) % 2**32,
                    )
                    waveform.generate_data()
                    data = waveform.get_data()

                    save_path = os.path.join(class_dir, f"data_{i}.npy")
                    np.save(save_path, data)

                except Exception as e:
                    print(f"Error generating {mod} sample {i}: {e}")
                    self.error_occurred.emit(f"Error generating {mod} sample {i}: {e}")

                count += 1
                self.sample_progress.emit(count, total)

        self.generation_finished.emit(self.output_dir)
