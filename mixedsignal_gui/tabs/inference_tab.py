"""
Inference Results Tab — batch model evaluation with confusion matrix,
classification report, and multi-class ROC curves.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QFileDialog, QTabWidget, QMessageBox, QScrollArea,
)
from PySide6.QtCore import Qt, QEvent, QSettings
from PySide6.QtWidgets import QApplication
import os
import json as _json
import numpy as np
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
)

from mixedsignal_gui.widgets.wheel_filter import install_wheel_blocker

# Models that expect 2-channel IQ input (must stay in sync with trainer.py)
_IQ_MODELS = {"ResNet1DOptimized"}


# ── Theming helpers ──────────────────────────────────────────────────────

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


# ═════════════════════════════════════════════════════════════════════════
class InferenceResultsTab(QWidget):
    """Model inference and evaluation tab."""

    def __init__(self, dataset_manager=None):
        super().__init__()
        self.dataset_manager = dataset_manager

        # ── state ──
        self.model = None
        self.model_path = None
        self.model_metadata = {}          # full JSON sidecar
        self.class_labels: list[str] = []
        self.model_in_channels = 1
        self.model_signal_length = 0      # 0 = unknown / use data length

        self.eval_data = None             # (N, C, L) float32 tensor
        self.eval_labels = None           # (N,) int64 array

        # Cached predictions (invalidated when model or data change)
        self._cached_y_pred = None        # (N,) int
        self._cached_probs = None         # (N, K) float

        # Cached plot data for theme redraws
        self._cm_data = None
        self._roc_data = None

        self._setup_ui()
        install_wheel_blocker(self)

    # ── Theme change ─────────────────────────────────────────────────────

    def changeEvent(self, event):
        if event.type() == QEvent.Type.PaletteChange:
            _theme_toolbar(self.cm_toolbar)
            _theme_toolbar(self.roc_toolbar)
            if self._cm_data:
                self._plot_confusion_matrix(*self._cm_data)
            if self._roc_data:
                self._plot_roc_curves(*self._roc_data)
        super().changeEvent(event)

    # ── UI setup ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left panel — configuration / controls (scrollable)
        left_panel = self._create_configuration_panel()
        left_scroll = QScrollArea()
        left_scroll.setObjectName("card")
        left_scroll.setWidget(left_panel)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(left_scroll, 1)

        # Right panel — visualizations
        right_panel = self._create_visualizations_panel()
        layout.addWidget(right_panel, 2)

    def _create_configuration_panel(self):
        """Create the left-side controls panel (model + data loading)."""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)

        # Header
        title = QLabel("Model Evaluation")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Load a trained model and evaluate on test data")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # ── Model section ────────────────────────────────────────────────
        model_label = QLabel("Model")
        model_label.setProperty("class", "section-title")
        layout.addWidget(model_label)

        self.load_model_btn = QPushButton("Load Model (.pth)")
        self.load_model_btn.clicked.connect(self._on_load_model)
        layout.addWidget(self.load_model_btn)

        self.model_label = QLabel("No model loaded")
        self.model_label.setProperty("class", "stat-label")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)

        layout.addWidget(QLabel("Classes:"))
        self.class_labels_label = QLabel("—")
        self.class_labels_label.setProperty("class", "stat-label")
        self.class_labels_label.setWordWrap(True)
        layout.addWidget(self.class_labels_label)

        # ── Data section ─────────────────────────────────────────────────
        data_label = QLabel("Test Data")
        data_label.setProperty("class", "section-title")
        layout.addWidget(data_label)

        self.load_data_btn = QPushButton("Load Test Data Folder")
        self.load_data_btn.clicked.connect(self._on_load_test_folder)
        self.load_data_btn.setEnabled(False)
        layout.addWidget(self.load_data_btn)

        self.load_registry_btn = QPushButton("Load from Dataset Registry")
        self.load_registry_btn.clicked.connect(self._on_load_from_registry)
        self.load_registry_btn.setEnabled(False)
        layout.addWidget(self.load_registry_btn)

        self.data_label = QLabel("No test data loaded")
        self.data_label.setProperty("class", "stat-label")
        self.data_label.setWordWrap(True)
        layout.addWidget(self.data_label)

        # ── Evaluate button ──────────────────────────────────────────────
        self.eval_all_btn = QPushButton("▶ Evaluate All")
        self.eval_all_btn.setEnabled(False)
        self.eval_all_btn.clicked.connect(self._evaluate_all)
        self.eval_all_btn.setMinimumHeight(36)
        layout.addWidget(self.eval_all_btn)

        # ── Classification report (text) ─────────────────────────────────
        report_title = QLabel("Classification Report")
        report_title.setProperty("class", "section-title")
        layout.addWidget(report_title)

        self.report_label = QLabel("Run evaluation to see report")
        self.report_label.setProperty("class", "stat-label")
        self.report_label.setWordWrap(True)
        self.report_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.report_label)

        layout.addStretch()
        return panel

    def _create_visualizations_panel(self):
        """Create the right-side visualization panel (plots in tabs)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)

        self.eval_tabs = QTabWidget()

        # Confusion matrix tab
        cm_w = QWidget()
        cm_l = QVBoxLayout(cm_w)
        cm_l.setContentsMargins(0, 0, 0, 0)
        bg, _, _ = _mpl_theme()
        self.cm_figure = Figure(dpi=100, facecolor=bg)
        self.cm_canvas = FigureCanvas(self.cm_figure)
        self.cm_canvas.setStyleSheet("background: transparent;")
        self.cm_toolbar = NavigationToolbar(self.cm_canvas, cm_w)
        _theme_toolbar(self.cm_toolbar)
        cm_l.addWidget(self.cm_toolbar)
        cm_l.addWidget(self.cm_canvas, 1)
        self.eval_tabs.addTab(cm_w, "Confusion Matrix")

        # ROC curves tab
        roc_w = QWidget()
        roc_l = QVBoxLayout(roc_w)
        roc_l.setContentsMargins(0, 0, 0, 0)
        self.roc_figure = Figure(dpi=100, facecolor=bg)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        self.roc_canvas.setStyleSheet("background: transparent;")
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, roc_w)
        _theme_toolbar(self.roc_toolbar)
        roc_l.addWidget(self.roc_toolbar)
        roc_l.addWidget(self.roc_canvas, 1)
        self.eval_tabs.addTab(roc_w, "ROC Curves")

        layout.addWidget(self.eval_tabs)
        return panel

    # ── Device helper ────────────────────────────────────────────────────

    def _device(self):
        settings = QSettings("MyCompany", "MixedSignalGUI")
        mode = settings.value("mode", "CPU")
        if mode == "GPU" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ── Model loading ────────────────────────────────────────────────────

    def on_trained_model_ready(self, model_path: str, labels: list):
        """Slot: auto-load model emitted by MLTrainingTab after training."""
        self.class_labels = list(labels)
        self.class_labels_label.setText(", ".join(labels) if labels else "—")
        self._load_model(model_path, num_classes=len(labels) or 2)
        self._enable_data_buttons()

    def _on_load_model(self):
        """Manual model load via file dialog."""
        settings = QSettings("MyCompany", "MixedSignalGUI")
        default_dir = settings.value("modelPath", "")
        if not default_dir or not os.path.isdir(default_dir):
            default_dir = os.path.expanduser("~")

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", default_dir,
            "PyTorch Models (*.pth);;All Files (*)",
        )
        if not path:
            return

        # Read companion JSON sidecar
        sidecar = os.path.splitext(path)[0] + ".json"
        if os.path.exists(sidecar):
            try:
                with open(sidecar, encoding="utf-8") as f:
                    info = _json.load(f)
                self.class_labels = info.get("class_labels", [])
                self.class_labels_label.setText(
                    ", ".join(self.class_labels) if self.class_labels else "—"
                )
            except Exception:
                pass

        num_classes = len(self.class_labels) if self.class_labels else 2
        self._load_model(path, num_classes=num_classes)
        self._enable_data_buttons()

    def _load_model(self, filepath: str, num_classes: int = 2):
        """Load model weights, read metadata, update UI."""
        try:
            from mixedsignal_gui.backend.torch_models import get_model, external_plugin_path

            device = self._device()

            # Metadata sidecar
            meta_path = os.path.splitext(filepath)[0] + ".json"
            model_name = "SimpleCNN"
            signal_length = 0
            in_channels = 1
            # Architecture-shaping hyperparameters; empty for models saved
            # before they were recorded, which gives the library defaults
            # those models were built with.
            hparams = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = _json.load(f)
                    self.model_metadata = meta
                    model_name = meta.get("model_name", "SimpleCNN")
                    signal_length = int(meta.get("signal_length", 0))
                    in_channels = int(meta.get("input_channels", 1))
                    hparams = meta.get("model_hparams") or {}
                    if meta.get("num_classes"):
                        num_classes = int(meta["num_classes"])
                    if meta.get("class_labels"):
                        self.class_labels = meta["class_labels"]
                        self.class_labels_label.setText(", ".join(self.class_labels))
                except Exception as exc:
                    print(f"[InferenceTab] Warning reading metadata: {exc}")

            plugin_path = external_plugin_path(filepath, self.model_metadata)

            try:
                state_dict = torch.load(filepath, map_location=device, weights_only=True)
            except TypeError:
                state_dict = torch.load(filepath, map_location=device)

            self.model = get_model(
                model_name,
                num_classes=num_classes,
                input_size=signal_length or 256,
                in_channels=in_channels,
                plugin_path=plugin_path,
                **hparams,
            )
            self.model.load_state_dict(state_dict)
            self.model.to(device)
            self.model.eval()

            self.model_path = filepath
            self.model_in_channels = in_channels
            self.model_signal_length = signal_length
            self._invalidate_cache()

            dev_name = "GPU" if device == "cuda" else "CPU"
            self.model_label.setText(
                f"Loaded: {os.path.basename(filepath)} ({model_name}) on {dev_name}"
            )
        except Exception as exc:
            self.model_label.setText(f"Failed to load: {exc}")
            print(f"[InferenceTab] Error loading model: {exc}")

    def _enable_data_buttons(self):
        self.load_data_btn.setEnabled(self.model is not None)
        self.load_registry_btn.setEnabled(
            self.model is not None and self.dataset_manager is not None
        )

    # ── Data loading ─────────────────────────────────────────────────────

    def _on_load_from_registry(self):
        """Load signals from DatasetManager, grouped by modulation."""
        if self.dataset_manager is None or self.model is None:
            return

        entries = self.dataset_manager.scan()
        # Only use non-augmented entries that have an explicit modulation field
        base_entries = [
            e for e in entries
            if not e.get("augmented", False) and e.get("modulation")
        ]

        # This panel is labelled "Test Data", so it must not report accuracy on
        # samples the model trained on.  Batch Generate marks each sample
        # train/test; prefer the held-out ones, and say so when falling back to
        # everything, which is what this used to do unconditionally.
        held_out = [e for e in base_entries if e.get("data_split") == "test"]
        split_note = ""
        if held_out:
            base_entries = held_out
        else:
            split_note = " (no held-out test split found — these may include " \
                         "training samples)"

        if not base_entries:
            self.data_label.setText("No datasets with modulation metadata found")
            return

        # Build label map from model's class_labels, or infer from data
        if self.class_labels:
            label_map = {lbl: i for i, lbl in enumerate(self.class_labels)}
        else:
            mods = sorted({e["modulation"] for e in base_entries})
            label_map = {m: i for i, m in enumerate(mods)}
            self.class_labels = list(label_map.keys())
            self.class_labels_label.setText(", ".join(self.class_labels))

        X_list, y_list = [], []
        skipped = 0
        for entry in base_entries:
            mod = entry["modulation"]
            if mod not in label_map:
                skipped += 1
                continue
            try:
                sig = self.dataset_manager.load_signal(entry)
                arr = self._to_flat(sig)
                X_list.append(arr)
                y_list.append(label_map[mod])
            except Exception as exc:
                print(f"[InferenceTab] skip {entry.get('name','?')}: {exc}")
                skipped += 1

        if not X_list:
            self.data_label.setText("Failed to load any signals from registry")
            return

        self._build_eval_tensors(X_list, y_list)
        extra = f" ({skipped} skipped)" if skipped else ""
        self.data_label.setText(
            f"Loaded {len(X_list)} signals, {len(label_map)} classes{extra}{split_note}"
        )

    def _on_load_test_folder(self):
        """Load test data from a folder (files or class sub-folders)."""
        settings = QSettings("MyCompany", "MixedSignalGUI")
        default_dir = settings.value("dataPath", "")
        if not default_dir or not os.path.isdir(default_dir):
            default_dir = ""

        folder = QFileDialog.getExistingDirectory(
            self, "Select Test Data Folder", default_dir,
        )
        if not folder:
            return

        try:
            # Collect files
            signal_files = []
            for root, _dirs, files in os.walk(folder):
                for fname in files:
                    if fname.lower().endswith((".npy", ".npz", ".csv")):
                        signal_files.append(os.path.join(root, fname))
            if not signal_files:
                self.data_label.setText("No .npy / .npz / .csv files found")
                return

            label_map = (
                {lbl: i for i, lbl in enumerate(self.class_labels)}
                if self.class_labels else {}
            )
            inferred: list[str] = []
            unmatched: dict = {}

            X_list, y_list = [], []
            for fpath in sorted(signal_files):
                mod = self._infer_label(fpath, known=set(label_map) or None)
                if mod in label_map:
                    idx = label_map[mod]
                elif label_map:
                    # A model is loaded and this file belongs to none of its
                    # classes.  It used to be given index len(label_map)+k — a
                    # true label outside the model's output range, which the
                    # model can never predict, so it silently depressed the
                    # reported accuracy and added a phantom row.  Skip it and
                    # say so instead.
                    unmatched[mod] = unmatched.get(mod, 0) + 1
                    continue
                else:
                    if mod not in inferred:
                        inferred.append(mod)
                    idx = inferred.index(mod)

                arr = self._load_array(fpath)
                if arr is not None:
                    X_list.append(self._to_flat(arr))
                    y_list.append(idx)

            if inferred and not self.class_labels:
                self.class_labels = inferred
                self.class_labels_label.setText(", ".join(self.class_labels))

            if not X_list:
                detail = (f" — none matched the model's classes "
                          f"({', '.join(sorted(unmatched))})" if unmatched else "")
                self.data_label.setText(f"No loadable signals found{detail}")
                return

            self._build_eval_tensors(X_list, y_list)
            note = ""
            if unmatched:
                n = sum(unmatched.values())
                note = (f"; skipped {n} file(s) whose class is not in the model: "
                        f"{', '.join(sorted(unmatched))}")
            self.data_label.setText(
                f"Loaded {len(X_list)} signals from folder{note}")

        except Exception as exc:
            self.data_label.setText(f"Error: {exc}")
            print(f"[InferenceTab] Error: {exc}")

    # ── Data helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_flat(sig) -> np.ndarray:
        """Flatten a signal, preserving complex dtype.

        This used to interleave complex input as [I0, Q0, I1, Q1, ...] to pair
        with an even/odd split further down.  That round-tripped for complex
        data but meant real passband signals were split the same way, which is
        meaningless for them.  Complex is now carried through as complex and
        split into channels once, in _prepare_iq.
        """
        arr = np.asarray(sig)
        # Keep a multi-antenna capture 2-D.  Flattening it here concatenated
        # the antennas end to end, so truncating to the model length left only
        # antenna 0 and the rest of the capture was silently discarded.
        if arr.ndim == 2 and arr.shape[0] <= 64:
            return arr if np.iscomplexobj(arr) else arr.astype(np.float32)
        arr = arr.ravel()
        return arr if np.iscomplexobj(arr) else arr.astype(np.float32)

    @staticmethod
    def _infer_label(path: str, known=None) -> str:
        """Guess the class label for *path*, preferring the model's own classes.

        Splitting the filename on "_" and taking one token cannot represent a
        class whose name contains an underscore: ``test_5G_NR_4_....npy`` came
        back as ``5G``, a class no model has, which then became a phantom
        column in the confusion matrix and shifted every label beside it.

        *known* is the set of classes the loaded model can actually predict.
        When it is supplied, the parent folder and then the longest matching
        class name win, so ``5G_NR`` stays whole.
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        folder = os.path.basename(os.path.dirname(path))

        if known:
            if folder in known:
                return folder
            body = stem
            for prefix in ("test_", "train_"):
                if body.lower().startswith(prefix):
                    body = body[len(prefix):]
                    break
            # Longest first, so 5G_NR beats a hypothetical 5G.
            for cand in sorted(known, key=len, reverse=True):
                if body == cand or body.startswith(cand + "_"):
                    return cand

        parts = stem.split("_")
        # No model classes to match against: strip an optional test_/train_
        # prefix and the trailing <M>_<date>_<time>_<micros> fields, keeping
        # the name tokens, so an underscored class survives here too.
        if parts and parts[0].lower() in ("test", "train"):
            parts = parts[1:]
        while len(parts) > 1 and parts[-1].isdigit():
            parts.pop()
        return "_".join(parts) if parts and parts[0] else folder

    @staticmethod
    def _load_array(path: str):
        try:
            if path.lower().endswith((".npy", ".npz")):
                arr = np.load(path, allow_pickle=True)
                if isinstance(arr, np.lib.npyio.NpzFile):
                    arr = arr[list(arr.keys())[0]]
                return np.asarray(arr)
            if path.lower().endswith(".csv"):
                return np.loadtxt(path, delimiter=",")
        except Exception as exc:
            print(f"[InferenceTab] Cannot load {path}: {exc}")
        return None

    def _prepare_iq(self, X: np.ndarray) -> np.ndarray:
        """Split complex baseband (N, L) into (N, 2, L) I/Q channels.

        Requires complex input.  This used to split even/odd samples of a real
        array, which only makes sense for interleaved [I0,Q0,I1,Q1,...] files —
        not for the real passband signals this app produces by default, where
        consecutive samples are two decimated copies of the same carrier.
        """
        if not np.iscomplexobj(X):
            raise ValueError(
                "This model expects 2-channel I/Q, but the test data is real "
                "(passband). Evaluate it with a model trained on real data, or "
                "load baseband (complex) test signals.")
        return np.stack([X.real, X.imag], axis=1).astype(np.float32)

    @staticmethod
    def _normalize_iq(X_iq: np.ndarray) -> np.ndarray:
        power = np.mean(X_iq[:, 0] ** 2 + X_iq[:, 1] ** 2, axis=1, keepdims=True)
        scale = np.sqrt(np.maximum(power, 1e-10))[:, np.newaxis, :]
        return X_iq / scale

    def _build_eval_tensors(self, X_list: list, y_list: list):
        """Pad/truncate, shape for the model, store as tensors."""
        # Target length: use model's expected length, or max in batch.
        # shape[-1], not .size — a (4, 2048) capture is 2048 long, not 8192.
        target_len = self.model_signal_length or max(a.shape[-1] for a in X_list)

        # Multi-antenna captures go through the same packing the trainer used,
        # so the channels mean the same thing on both sides.
        if any(np.asarray(a).ndim == 2 for a in X_list):
            from mixedsignal_gui.backend.trainer import pack_multichannel
            X = np.stack([pack_multichannel(a, target_len) for a in X_list])
            self.eval_data = torch.from_numpy(X)
            self.eval_labels = np.asarray(y_list, dtype=np.int64)
            self._invalidate_cache()
            self.eval_all_btn.setEnabled(True)
            self.eval_tabs.setEnabled(True)
            return

        # Keep the buffer complex when the data is, so Q survives to _prepare_iq.
        any_complex = any(np.iscomplexobj(a) for a in X_list)
        X = np.zeros((len(X_list), target_len),
                     dtype=np.complex64 if any_complex else np.float32)
        for i, a in enumerate(X_list):
            L = min(len(a), target_len)
            X[i, :L] = a[:L]

        if self.model_in_channels == 2:
            X = self._prepare_iq(X)
            X = self._normalize_iq(X)
        else:
            # np.real is a no-op on real data and takes I from complex input.
            X = np.real(X).astype(np.float32)[:, np.newaxis, :]   # (N, 1, L)

        self.eval_data = torch.from_numpy(X)
        self.eval_labels = np.asarray(y_list, dtype=np.int64)
        self._invalidate_cache()

        self.eval_all_btn.setEnabled(True)
        self.eval_tabs.setEnabled(True)

    # ── Cached inference ─────────────────────────────────────────────────

    def _invalidate_cache(self):
        self._cached_y_pred = None
        self._cached_probs = None

    def _run_inference(self):
        """Run the model on eval_data once and cache predictions + probs."""
        if self._cached_y_pred is not None:
            return True   # already cached

        if self.model is None or self.eval_data is None:
            return False

        try:
            device = self._device()
            self.model.to(device)
            data = self.eval_data.to(device)

            with torch.no_grad():
                outputs = self.model(data)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = outputs.max(1)

            self._cached_y_pred = preds.cpu().numpy()
            self._cached_probs = probs.cpu().numpy()
            return True

        except Exception as exc:
            QMessageBox.warning(self, "Inference Error", str(exc))
            print(f"[InferenceTab] Inference error: {exc}")
            return False

    # ── Evaluate All ─────────────────────────────────────────────────────

    def _evaluate_all(self):
        if not self._run_inference():
            return
        self._show_confusion_matrix()
        self._show_report()
        self._show_roc()

    # ── Confusion matrix ─────────────────────────────────────────────────

    def _show_confusion_matrix(self):
        if self._cached_y_pred is None:
            return
        # Pin the axes to the model's own classes.  Without labels=, sklearn
        # sizes the matrix from whatever labels happen to be present, so a test
        # set missing a class silently shifted every row and column against the
        # names drawn beside them.
        cm = confusion_matrix(self.eval_labels, self._cached_y_pred,
                              labels=list(range(self._num_classes())))
        self._cm_data = (cm,)
        self._plot_confusion_matrix(cm)

    def _num_classes(self) -> int:
        """How many classes the loaded model can actually predict."""
        if self.class_labels:
            return len(self.class_labels)
        return int(self.model_metadata.get("num_classes", 0)) or (
            int(max(self.eval_labels)) + 1 if self.eval_labels is not None else 0)

    def _plot_confusion_matrix(self, cm):
        self.cm_figure.clear()
        ax = self.cm_figure.add_subplot(111)

        n = cm.shape[0]
        labels = (
            self.class_labels if len(self.class_labels) == n
            else [str(i) for i in range(n)]
        )

        # Scale font sizes with number of classes
        cell_fs = max(7, 14 - n)       # annotation font inside cells
        tick_fs = max(7, 12 - n // 2)  # tick-label font

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest", aspect="equal")
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, pad=10)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=tick_fs)
        ax.set_yticklabels(labels, fontsize=tick_fs)

        # Cell annotations
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=cell_fs, fontweight="bold")

        cbar = self.cm_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _, fg, _ = _mpl_theme()
        cbar.ax.yaxis.set_tick_params(color=fg)
        for lbl in cbar.ax.get_yticklabels():
            lbl.set_color(fg)

        _apply_mpl_theme(self.cm_figure, ax)
        self.cm_figure.tight_layout(rect=[0, 0, 0.95, 1])
        self.cm_canvas.draw()

    # ── Classification report ────────────────────────────────────────────

    def _show_report(self):
        if self._cached_y_pred is None:
            return
        # labels= must accompany target_names, or sklearn raises as soon as the
        # test set does not contain every class ("Number of classes, 2, does
        # not match size of target_names, 4"), which left the report showing
        # its previous text beside a freshly drawn matrix.
        n = self._num_classes()
        target_names = self.class_labels if len(self.class_labels) == n else None
        report = classification_report(
            self.eval_labels, self._cached_y_pred,
            labels=list(range(n)),
            target_names=target_names, zero_division=0,
        )
        accuracy = (self._cached_y_pred == self.eval_labels).mean()
        self.report_label.setText(f"Accuracy: {accuracy:.4f}\n\n{report}")

    # ── ROC curves (one-vs-rest for multi-class) ─────────────────────────

    def _show_roc(self):
        if self._cached_probs is None:
            return

        K = self._cached_probs.shape[1]
        labels = (
            self.class_labels if len(self.class_labels) == K
            else [str(i) for i in range(K)]
        )

        roc_data: list[tuple] = []  # [(label, fpr, tpr, auc_val), ...]

        for k in range(K):
            y_bin = (self.eval_labels == k).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue                          # skip if class absent
            fpr, tpr, _ = roc_curve(y_bin, self._cached_probs[:, k])
            auc_val = auc(fpr, tpr)
            roc_data.append((labels[k], fpr, tpr, auc_val))

        if not roc_data:
            self.report_label.setText(
                self.report_label.text() + "\n\n(ROC: not enough classes with data)"
            )
            return

        self._roc_data = (roc_data,)
        self._plot_roc_curves(roc_data)

    def _plot_roc_curves(self, roc_data):
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)

        cmap = plt_cmap(len(roc_data))
        for i, (label, fpr, tpr, auc_val) in enumerate(roc_data):
            ax.plot(fpr, tpr, lw=2, color=cmap(i),
                    label=f"{label} (AUC={auc_val:.2f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, pad=10)
        ax.legend(loc="lower right", fontsize=8)

        _apply_mpl_theme(self.roc_figure, ax)
        self.roc_figure.tight_layout()
        self.roc_canvas.draw()


# ── Utility ──────────────────────────────────────────────────────────────

def plt_cmap(n: int):
    """Return a callable colormap with *n* distinct colours.

    Uses ``matplotlib.colormaps``; ``plt.cm.get_cmap`` was deprecated in 3.7
    and removed in 3.9, so on a current matplotlib this raised AttributeError
    on every evaluation.  Nothing caught it, so the ROC tab came up blank while
    the confusion matrix and report beside it looked healthy.
    """
    import matplotlib
    cm = matplotlib.colormaps["tab10" if n <= 10 else "tab20"]
    return lambda i: cm(i % cm.N)
