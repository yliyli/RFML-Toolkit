# RFML Toolkit

**Signal Generation & Classification Dashboard**

A PySide6-based application connecting waveform generation, channel-aware augmentation, PyTorch training, and evaluation on generated or recorded I/Q. Interchangeable waveform, channel, and learning components share NumPy samples and experiment metadata for reproducible RFML experimentation.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **Waveform Selection** – Generate individual samples or named datasets using Python and MATLAB waveform generators, including LoRa, Zigbee, and 5G NR.
- **Channel & Noise** – Preview or batch-apply stochastic channels, Sionna ray tracing with bundled digital twins, or imported measured-response banks, with optional noise.
- **ML Training** – Train built-in PyTorch architectures or trusted user-supplied Python plugins. Configure the input sample length (default 2048); checkpoints preserve class labels and preprocessing metadata.
- **Inference Results / Evaluate Model** – Reload checkpoints, evaluate labeled datasets, or classify individual generated/imported samples; inspect confusion matrices, reports, ROC curves, and waveform views.
- **Data Visualization** – Explore loaded datasets and export visualization coordinates and plots.
- **Dataset Management & Exports** – Store named generated and augmented datasets as `.npy`/`.json` pairs, import external class folders or annotated SigMF recordings, and export plots/CSV results grouped by model/data pair.

---

## Installation

### Quick Install

For the current features described below, install from the repository:

```bash
git clone https://github.com/yliyli/RFML-Toolkit.git
cd RFML-Toolkit
pip install -e .
```

### Prerequisites

- **Python 3.10 or higher** (the current local test environment uses 3.12; backend compatibility depends on installed releases)
- **(Optional) MATLAB** – For advanced waveform generation features
  - [MATLAB](https://www.mathworks.com/products/matlab.html) with the toolboxes required by the selected waveform family
  - [MATLAB Engine for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

### Installing MATLAB Engine (Optional)

For MATLAB-based generation, follow the
[MATLAB Engine installation instructions](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
for your installed MATLAB release and a supported Python version. Install the
Engine into the same environment used to launch the GUI, then verify it:

```bash
python -c "import matlab.engine; print('MATLAB Engine import succeeded')"
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

* Provides an introductory workflow tour; the current application has six tabs:
  Waveform Selection, Channel & Noise, ML Training, Inference Results,
  Evaluate Model, and Data Visualization
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

On ML Training, **Load Toolbox Datasets** opens a folder picker starting at the
Data folder in Settings. Choose the generated or augmented folder to load its
direct .npy/JSON pairs, using metadata class labels and excluding held-out test
entries. Load additional folders to merge examples under the same class names;
already-loaded files are skipped. Subfolders are not automatically included.
Generation writes into **Dataset Folder Name** beneath the Settings Data folder.
Augmentation writes into **Augmentation Folder Name** beneath that same root,
beside the base dataset—not inside the selected source folder. Loading a folder
for training does not change either save destination.
**Import External Data** selects a parent folder with class-named subfolders:
one `.npy`, `.npz`, or `.csv` per example. Complex IQ has shape `(N,)`;
`(antennas, N)` denotes antenna channels, not a batch. Hover over the circled
question marks for dataset/model requirements.

To import observed I/Q, click **Import Long SigMF** on ML Training or
**Import SigMF Recording…** on Inference Results and select either member
of a `.sigmf-meta`/`.sigmf-data` pair.
Annotated sample regions (or whole captures when no annotations exist) are saved
unchanged as NumPy arrays with SigMF provenance in the shared dataset registry.
`core:label` becomes the training class and `core:frequency` supplies the capture
frequency. Use **Load Toolbox Datasets** for labeled training regions, or **Load from
Dataset Registry** for evaluation. Unlabeled regions remain stored without an
invented class. In **Evaluate Model → Import a Sample & Classify**, select a SigMF file and
choose one region for plotting and single-waveform classification.
Import does not automatically window a recording. Define labeled annotation
regions with `core:sample_start`, `core:sample_count`, and `core:label` before
import to obtain multiple examples. Training truncates/pads each example to
the selected **Sample Length** (default 2048); native stored samples remain unchanged.
Evaluation displays and uses the saved checkpoint length. Keep windows from the
same recording in the same train/test partition to avoid leakage.

Generation saves into the editable **Dataset Folder Name** under the Settings
Data folder (default `generated_<timestamp>`). Reusing the name adds examples to
that folder. Channel saves use **Augmentation Folder Name** (default
`augmented_<timestamp>`): both bulk and single-example output are under the
Settings Data folder, beside generated dataset folders. Bulk runs require a
new output folder name. Channel preview discovers examples in these subfolders;
training loads the specific folder selected with **Load Toolbox Datasets**.

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

Training power-normalizes every example after padding/truncation, for real,
complex I/Q, and multi-antenna inputs. One scale per example sets mean antenna
power to one while preserving relative antenna levels. Zero inputs remain zero.
New checkpoints record this rule under `power_normalization`, and both inference
views reuse it. Older checkpoints without that field retain their original
preprocessing; retrain them to use the new rule.

LoRa chirps are generated directly in Python, even when MATLAB is available.
`M` selects spreading factor 7–12 (clamped to that range); `fs * Tsymb`, rounded
to an integer and at least two, sets samples per chip and hence chirp bandwidth.
The packet contains preamble, sync, down-chirp delimiter, and randomized payload
chirps, then is cropped to the requested length using the experiment seed.
Baseband I/Q and real passband outputs are supported. This is waveform synthesis,
not a complete LoRa protocol encoder.

### Result exports

**Export Results** in Inference Results, Evaluate Model (including Channel Test),
and Data Visualization writes PNG plots and numeric CSV tables to
**Settings → Paths → Output folder**. The default is `~/Documents/rfml_output`.
Export is enabled only after a successful evaluation or visualization.

Each model/data pair has a content-fingerprinted folder; repeated exports create
new `run_<UTC timestamp>` subfolders with `plots/`, `csv/`, and `run.txt`/`run.json`
provenance. The root `index.txt` documents all pairs and their run counts.
Fingerprints include loaded weights and model metadata, native sample contents,
ordered labels, and class names—not just filenames. Visualization uses a
dataset-only group, without a model. Changes to the model, labels, or data form a
new group. CSVs contain predictions/probabilities, confusion matrices/reports/ROC,
single-waveform samples/model inputs, or visualization coordinates as applicable.
Exports do not modify samples or recalibrate plot units; no data is exported
automatically, and previous runs are never overwritten. Reload test data after
changing the batch-evaluation checkpoint to apply its input preprocessing.

### MATLAB Integration

If MATLAB and the MATLAB Engine are installed, the app will automatically:

- Start the MATLAB engine
- Add `waveform_functions` to the MATLAB path
- Enable MATLAB-based waveform generators

Without MATLAB, Python generation supports PAM, QAM, PSK, FSK, FHSS, LFM,
Barker, FMCW, and LoRa. WiFi, LTE, 5G NR, and Zigbee require MATLAB and the
appropriate toolboxes. LoRa uses Python even when MATLAB is available.
Standards-based generators contain fixed PHY configurations; generic `M` and
`Tsymb` controls do not change every family. Zigbee uses the 2.4-GHz OQPSK PHY
with a 2-Mchip/s chip rate.

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
│   ├── torch_models.py
│   ├── trainer.py
│   ├── waveform_pipeline.py
│   └── waveform_service.py
├── sionna_widget/              # Custom widgets for Sionna library
├── styles/                     # Stylesheet definitions
├── tabs/                       # UI tabs for each workflow step
├── waveform_functions/         # MATLAB scripts (added to path at runtime)
└── resources/                  # UI resources and assets
```

Dataset, checkpoint, and exported-result destinations are configured in
**Settings → Paths**; they do not need to live inside the package directory.

---

## Development

### Adding New Features

- **Stylesheets**: Customize in `styles/stylesheet.py`
- **Waveform generators**: Add to `backend/generators.py` or as MATLAB scripts in `waveform_functions/`
- **Dataset management**: Configure in `backend/dataset_manager.py`
- **New UI tabs**: Follow patterns in `tabs/` directory

### Running Tests

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -q
```

Native scene rendering and hover interactions also need a desktop smoke check;
headless tests are not a guarantee against GPU/OpenGL driver failures.

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
- TensorFlow – Backend for channel components (classifier training uses PyTorch)
- Sionna / Sionna RT – Communications and ray-tracing backends
- NumPy – Numerical computing
- Matplotlib – Plotting library
- scikit-learn – Machine learning utilities
- PyOpenGL – OpenGL bindings

See `pyproject.toml` for declared dependencies. UMAP visualization additionally
requires `umap-learn`. MATLAB Engine and MATLAB toolboxes are installed separately;
backend releases and hardware support must be compatible with your environment.

---

## Troubleshooting

### MATLAB Engine Issues

**Error: "MATLAB engine is not available"**
- Ensure MATLAB is installed and the engine is installed in your Python environment
- Verify installation: `python -c "import matlab.engine; print('Success')"`
- Reinstall the engine if needed (see Installation section above)

### GPU Not Detected

The current training/inference device selector uses NVIDIA CUDA when **GPU** is
selected and CUDA is available; otherwise it falls back to CPU. It does not yet
select Apple's MPS backend, so choosing GPU on a Mac does not enable Metal
acceleration. TensorFlow/Sionna device availability is a separate backend check.

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

The camera-ready paper acknowledges NSF support under awards 2431961, 2526493,
and 2112471, and OpenStreetMap contributors (ODbL) for map-derived scene data.

This project makes use of:

- [PySide6](https://doc.qt.io/qtforpython/) – Python bindings for Qt
- [PyTorch](https://pytorch.org/) – Deep learning framework
- [TensorFlow](https://www.tensorflow.org/) – Machine learning platform
- [Sionna](https://sionna.readthedocs.io/) – Link-level communications simulator
- [scikit-learn](https://scikit-learn.org/) – Machine learning library

---

## Citation

If you use this toolbox in research, cite the accompanying paper. This entry
describes the camera-ready manuscript for the NRDZCOM8 workshop at IEEE MILCOM
2026; add the published proceedings details and DOI when available.

```bibtex
@unpublished{li2026rfmltoolbox,
  author = {Li, Yiyang and Borda, Nicholas and Chae, Tony and Deng, Christopher
            and Jayr, Amaury and Madan, Shivansh and Maldei-Stumm, Nastasia
            and Xu, Matthew and Prabhu, Agastya and Chowdhury, Kaushik},
  title = {An End-to-End {RFML} Toolbox for Channel-Aware Experimentation},
  year = {2026},
  note = {Camera-ready manuscript, NRDZCOM8 workshop at IEEE MILCOM 2026},
  url = {https://github.com/yliyli/RFML-Toolkit}
}
```

---

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/yliyli/RFML-Toolkit).
