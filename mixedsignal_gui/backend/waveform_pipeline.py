import numpy as np
from scipy import signal

from mixedsignal_gui.backend.core import Waveform
from mixedsignal_gui.backend.experiment_settings import experiment_seed


class WaveformPipeline:
    def __init__(self, matlab_engine, seed=None):
        self.matlab_engine = matlab_engine
        self.run_seed = experiment_seed() if seed is None else int(seed)
        self.entry_index = 0

    def generate(self, *, fs, Tsymb, Nsymb, fc, M, modulation,
                 var, alpha, span, pulse_shape, output_type="baseband"):
        entry_index = self.entry_index
        seed = (self.run_seed + entry_index) % 2**32
        self.entry_index += 1
        waveform = Waveform(
            fs=fs,
            Tsymb=Tsymb,
            Nsymb=Nsymb,
            fc=fc,
            M=M,
            modulation=modulation,
            var=var,
            matlab_engine=self.matlab_engine,
            alpha=alpha,
            span=span,
            pulse_shape=pulse_shape,
            output_type=output_type,
            seed=seed,
        )

        waveform.generate_data()
        data = waveform.get_data()
        sps = waveform.get_sps()

        T = len(data) / fs
        t = np.linspace(0, T, len(data))

        # Compute spectrum in Python (plotspec_gui is just fft+fftshift
        # and cannot accept complex numpy arrays via the MATLAB engine API)
        N = len(data)
        ft = np.fft.fftshift(np.fft.fft(data))
        freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))

        return {
            "time": t,
            "signal": data,
            "freqs": freqs,
            "spectrum": np.abs(ft),
            "sps": sps,
            "baseband_symbols": waveform.metadata.get("baseband_symbols"),
            # "matlab" or "python" — recorded in dataset metadata so a mixed
            # datasets/ folder stays traceable to how each signal was made.
            "generator": waveform.metadata.get("generator"),
            "seed_metadata": {"run_seed": self.run_seed, "entry_index": entry_index,
                              "generation_seed": seed},
        }
