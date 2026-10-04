# RFML Toolkit

**Signal Generation & Classification Dashboard**

A PySide6-based graphical application for generating digital communication waveforms, applying channel models and noise, training machine learning models, and performing inference.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Waveform Selection** – Choose modulation types (FSK, PAM, FHSS, etc.) and parameters. Generation routines can call MATLAB functions when the MATLAB Engine API is available.
- **Channel & Noise** – Apply channel models and additive noise to generated signals.
- **ML Training** – Train classification models using TensorFlow or PyTorch on synthetic waveform datasets.
- **Inference Results** – Visualize predictions, confusion matrices, and performance metrics.
- **Dataset Management** – Save/load examples in `.npy`/`.json` pairs using a shared `DatasetManager`.

---

## Installation

### Quick Install

```bash
pip install rfml-toolkit
```

### Prerequisites

- **Python 3.10 or higher** (tested with 3.11)
- **(Optional) MATLAB** – For advanced waveform generation features
  - [MATLAB](https://www.mathworks.com/products/matlab.html) R2021a or later
  - [MATLAB Engine for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

### Installing MATLAB Engine (Optional)

If you want to use MATLAB-based waveform generation:

1. **Find your MATLAB installation:**
   ```bash
   matlab -batch "disp(matlabroot)"
   ```

2. **Install the MATLAB Engine for Python:**
   ```bash
   # macOS/Linux
   cd /Applications/MATLAB_R2023b.app/extern/engines/python
   python setup.py install
   
   # Windows
   cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"
   python setup.py install
   ```

   Replace `R2023b` with your MATLAB version.

### Install from Source (for developers)

```bash
git clone https://github.com/yliyli/RFML-Toolkit.git
cd RFML-Toolkit
pip install -e .
```

For development dependencies:
```bash
pip install -e ".[dev]"
```

---

## Usage

After installation, launch the GUI from the command line:

```bash
rfml-toolkit
```

Or from Python:

```python
from mixedsignal_gui.main_window import main
main()
```

### First Launch

On the first launch, the application displays a multi-page setup wizard that:

* Provides a **quick tour** of the four main tabs (waveform, channel, ML, inference)
* Lets you select folders for models and datasets
* Detects available GPUs and allows CPU/GPU mode selection
* Shows tips & tricks for navigation

Your settings are saved and the wizard won't appear again unless you reset settings or access it from the Help menu.

### Reproducible runs and recorded I/Q

Set **Settings → Compute → Experiment seed** (default: 42) and click Apply or OK.
The value persists across restarts and is used by new generation, augmentation,
training, and randomized visualization runs. Bulk runs derive distinct per-example seeds, record the run
seed and entry index in each JSON sidecar, and process source names in a stable
order. Repeatability requires the same input files, settings, and software/backend.

To import observed I/Q, click **Import SigMF Recording…** on ML Training or
Inference Results and select either member of a `.sigmf-meta`/`.sigmf-data` pair.
Annotated sample regions (or whole captures when no annotations exist) are saved
unchanged as NumPy arrays with SigMF provenance in the shared dataset registry.
`core:label` becomes the training class and `core:frequency` supplies the capture
frequency. Use **Load from Datasets** for labeled training regions, or **Load from
Dataset Registry** for evaluation. Unlabeled regions remain stored without an
invented class. In **Evaluate Model → Import Waveform**, select a SigMF file and
choose one region for plotting and single-waveform classification.

The importer requires a positive `core:sample_rate` and supports single-channel
`cf32`, `cf64`, `ci8`, `ci16`, and `ci32` complex data, including declared little-
or big-endian formats. Headers, offsets, nonstandard data paths, and multichannel
recordings are rejected explicitly. Recordings are observed signals; the channel
bank's import controls are for separately extracted channel responses.

RT taps use the waveform's sampling rate independently of channel bandwidth.
Changing the dataset rate, bandwidth, or tap interval regrids the existing
paths before applying them. RT folder runs require a common sample rate;
process different-rate groups separately. A successful single-waveform apply
captures its samples, source metadata, channel configuration, and antenna save
mode together, so switching tabs or editing controls before saving does not
change the saved provenance.

### MATLAB Integration

If MATLAB and the MATLAB Engine are installed, the app will automatically:
- Start the MATLAB engine
- Add `waveform_functions` to the MATLAB path
- Enable MATLAB-based waveform generators

If MATLAB is not available, the GUI will still launch with Python-based waveform generation, but some features will be disabled.

---

## Project Structure

```
mixedsignal_gui/
├── main_window.py              # Entry point for the PySide6 application
├── backend/                    # Signal logic, dataset & model utilities
│   ├── augmentation.py
│   ├── core.py
│   ├── dataset_generator.py
│   ├── dataset_manager.py
│   ├── generators.py
│   ├── matlab_engine.py        # Wrapper for MATLAB Engine API
│   ├── tf_models.py
│   ├── torch_models.py
│   ├── trainer.py
│   ├── waveform_pipeline.py
│   └── waveform_service.py
├── sionna_widget/              # Custom widgets for Sionna library
├── styles/                     # Stylesheet definitions
├── tabs/                       # UI tabs for each workflow step
├── waveform_functions/         # MATLAB scripts (added to path at runtime)
├── resources/                  # UI resources and assets
├── datasets/                   # Generated examples and metadata
└── models/                     # Trained model snapshots
```

---

## Development

### Adding New Features

- **Stylesheets**: Customize in `styles/stylesheet.py`
- **Waveform generators**: Add to `backend/generators.py` or as MATLAB scripts in `waveform_functions/`
- **Dataset management**: Configure in `backend/dataset_manager.py`
- **New UI tabs**: Follow patterns in `tabs/` directory

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black mixedsignal_gui/
isort mixedsignal_gui/
```

---

## Dependencies

Core dependencies:
- PySide6 – Qt-based GUI framework
- PyTorch – Deep learning framework
- TensorFlow – Machine learning platform
- Sionna – Link-level simulator
- NumPy – Numerical computing
- Matplotlib – Plotting library
- scikit-learn – Machine learning utilities
- PyOpenGL – OpenGL bindings

See `pyproject.toml` for the complete list.

---

## Troubleshooting

### MATLAB Engine Issues

**Error: "MATLAB engine is not available"**
- Ensure MATLAB is installed and the engine is installed in your Python environment
- Verify installation: `python -c "import matlab.engine; print('Success')"`
- Reinstall the engine if needed (see Installation section above)

### GPU Not Detected

- For PyTorch: Check with `python -c "import torch; print(torch.cuda.is_available())"`
- For TensorFlow: Check with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- Ensure appropriate CUDA drivers are installed

### Import Errors

If you get "No module named X" errors, the package may be missing from dependencies. Please report these as issues.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This project makes use of:
- [PySide6](https://doc.qt.io/qtforpython/) – Python bindings for Qt
- [PyTorch](https://pytorch.org/) – Deep learning framework
- [TensorFlow](https://www.tensorflow.org/) – Machine learning platform
- [Sionna](https://sionna.readthedocs.io/) – Link-level communications simulator
- [scikit-learn](https://scikit-learn.org/) – Machine learning library

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{rfml_toolkit,
  author = {Maldei-Stumm, Nastasia},
  title = {RFML Toolkit: Signal Generation & Classification Dashboard},
  year = {2025},
  url = {https://github.com/yliyli/RFML-Toolkit}
}
```

---

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/yliyli/RFML-Toolkit).
