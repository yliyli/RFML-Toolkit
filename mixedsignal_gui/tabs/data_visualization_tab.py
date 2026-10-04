"""
tabs/data_visualization_tab.py

Data Visualization Tab — PCA / t-SNE / UMAP
Loads .npy waveform files from waveform_data/<modulation>/
and visualizes them in 2D and 3D scatter plots.
"""

import os
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox, QScrollArea,
    QProgressBar, QMessageBox, QCheckBox, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QEvent, QSettings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mixedsignal_gui.widgets.wheel_filter import install_wheel_blocker
from mixedsignal_gui.backend.experiment_settings import experiment_seed


# ─────────────────────────────────────────────
# Background worker thread
# ─────────────────────────────────────────────
class DimReduceThread(QThread):
    """Run dimensionality reduction off the main thread."""
    finished = Signal(object, object, list)   # coords_2d, coords_3d, labels
    error    = Signal(str)
    progress = Signal(str)

    def __init__(self, X, y, method, params, parent=None):
        super().__init__(parent)
        self.X      = X          # (N, D) float32
        self.y      = y          # (N,)  int  — class indices
        self.method = method     # "PCA" | "t-SNE" | "UMAP"
        self.params = params     # dict of hyper-parameters
        self.seed = experiment_seed()

    def run(self):
        try:
            X = self.X
            method = self.method
            p = self.params

            self.progress.emit(f"Running {method}…")

            if method == "PCA":
                from sklearn.decomposition import PCA
                coords_2d = PCA(n_components=2,
                                random_state=self.seed).fit_transform(X)
                coords_3d = PCA(n_components=3,
                                random_state=self.seed).fit_transform(X)

            elif method == "t-SNE":
                from sklearn.manifold import TSNE
                import sklearn
                perplexity = min(p.get("perplexity", 30), len(X) - 1)
                # sklearn >= 1.5 renamed n_iter → max_iter
                sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
                iter_kwarg = "max_iter" if sk_version >= (1, 5) else "n_iter"
                tsne_kwargs = dict(
                    perplexity=perplexity,
                    learning_rate=p.get("learning_rate", 200),
                    random_state=self.seed,
                )
                tsne_kwargs[iter_kwarg] = p.get("n_iter", 1000)
                coords_2d = TSNE(n_components=2, **tsne_kwargs).fit_transform(X)
                coords_3d = TSNE(n_components=3, **tsne_kwargs).fit_transform(X)

            elif method == "UMAP":
                import umap as umap_lib
                coords_2d = umap_lib.UMAP(
                    n_neighbors=p.get("n_neighbors", 15),
                    min_dist=p.get("min_dist", 0.1),
                    n_components=2,
                    metric=p.get("metric", "euclidean"),
                    random_state=self.seed
                ).fit_transform(X)
                coords_3d = umap_lib.UMAP(
                    n_neighbors=p.get("n_neighbors", 15),
                    min_dist=p.get("min_dist", 0.1),
                    n_components=3,
                    metric=p.get("metric", "euclidean"),
                    random_state=self.seed
                ).fit_transform(X)

            else:
                raise ValueError(f"Unknown method: {method}")

            self.finished.emit(coords_2d, coords_3d, self.y)

        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────────────────────────
# Matplotlib canvas helpers
# ─────────────────────────────────────────────
class ScatterCanvas2D(FigureCanvas):
    def __init__(self, parent=None):
        self._last_plot_data = None
        
        bg, _, _ = self._theme()
        self.fig = Figure(figsize=(5, 4), tight_layout=True, facecolor=bg)
        self.ax  = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: transparent;")
        
        self._draw_placeholder("2D Visualization")
    
    def changeEvent(self, event):
        if event.type() == QEvent.Type.PaletteChange:
            if self._last_plot_data:
                self.plot(*self._last_plot_data)
            else:
                self._draw_placeholder("2D Visualization")
        super().changeEvent(event)

    def _theme(self):
        """Return (bg, fg, axes_bg) from the Qt application palette."""
        from PySide6.QtWidgets import QApplication
        p = QApplication.palette()
        return (p.window().color().name(),
                p.windowText().color().name(),
                p.base().color().name())

    def _draw_placeholder(self, msg=""):
        bg, fg, axes_bg = self._theme()
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha="center", va="center",
                     fontsize=12, color=fg, transform=self.ax.transAxes)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(axes_bg)
        self.draw()

    def plot(self, coords, labels, class_names, title="2D", visible_classes=None):
        self._last_plot_data = (coords, labels, class_names, title, visible_classes)
        
        bg, fg, axes_bg = self._theme()
        self.ax.clear()
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(axes_bg)
        cmap = plt.get_cmap("tab20", len(class_names))
        for i, name in enumerate(class_names):
            if visible_classes is not None and name not in visible_classes:
                continue
            mask = np.array(labels) == i
            if mask.sum() == 0:
                continue
            self.ax.scatter(coords[mask, 0], coords[mask, 1],
                            s=12, alpha=0.7, color=cmap(i), label=name)
        self.ax.legend(fontsize=7, loc="best",
                       facecolor=axes_bg, labelcolor=fg,
                       markerscale=1.5)
        self.ax.set_title(title, color=fg, fontsize=11)
        self.ax.tick_params(colors=fg)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(fg)
        self.draw()


class ScatterCanvas3D(FigureCanvas):
    """3-D scatter with reduced rotation sensitivity and reset-view support."""

    _DEFAULT_ELEV = 20
    _DEFAULT_AZIM = 45

    def __init__(self, parent=None):
        self._last_plot_data = None
        
        bg, _, _ = self._theme()
        self.fig = Figure(figsize=(5, 4), tight_layout=True, facecolor=bg)
        self.ax  = self.fig.add_subplot(111, projection="3d")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setStyleSheet("background: transparent;")

        # ── reduce rotation sensitivity ──────────────────────────────
        # Matplotlib's 3-D axes use mouse_init() to bind drag events.
        # We override _on_move to scale the delta down.
        self._drag_start  = None   # (x, y, elev, azim) at button-press
        self.ax.mouse_init(rotate_btn=1, zoom_btn=None)   # disable zoom drag
        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event",  self._on_drag)

        self._last_coords      = None
        self._last_labels      = None
        self._last_class_names = []
        self._last_title       = "3D"

        self._draw_placeholder("3D Visualization")
    
    def changeEvent(self, event):
        if event.type() == QEvent.Type.PaletteChange:
            if self._last_plot_data:
                self.plot(*self._last_plot_data)
            else:
                self._draw_placeholder("3D Visualization")
        super().changeEvent(event)

    # ── custom rotation handlers ─────────────────────────────────────
    def _on_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self._drag_start = (event.x, event.y,
                                self.ax.elev, self.ax.azim)

    def _on_release(self, event):
        self._drag_start = None

    def _on_drag(self, event):
        if self._drag_start is None or event.inaxes != self.ax:
            return
        x0, y0, elev0, azim0 = self._drag_start
        dx = event.x - x0
        dy = event.y - y0
        SENSITIVITY = 0.25          # < 1  →  slower / more controlled
        new_azim = azim0 - dx * SENSITIVITY
        new_elev = elev0 - dy * SENSITIVITY
        new_elev = max(-85, min(85, new_elev))   # clamp: no full flip
        self.ax.view_init(elev=new_elev, azim=new_azim)
        self.draw_idle()

    # ── reset view ───────────────────────────────────────────────────
    def reset_view(self):
        self.ax.view_init(elev=self._DEFAULT_ELEV, azim=self._DEFAULT_AZIM)
        self.draw_idle()

    # ── drawing ──────────────────────────────────────────────────────
    def _theme(self):
        from PySide6.QtWidgets import QApplication
        p = QApplication.palette()
        return (p.window().color().name(),
                p.windowText().color().name(),
                p.base().color().name())

    def _draw_placeholder(self, msg=""):
        bg, fg, axes_bg = self._theme()
        self.ax.clear()
        self.ax.text2D(0.5, 0.5, msg, ha="center", va="center",
                       fontsize=12, color=fg, transform=self.ax.transAxes)
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(axes_bg)
        self.draw()

    def plot(self, coords, labels, class_names, title="3D", visible_classes=None):
        self._last_plot_data = (coords, labels, class_names, title, visible_classes)
        
        self._last_coords       = coords
        self._last_labels       = labels
        self._last_class_names  = class_names
        self._last_title        = title
        self._last_visible      = visible_classes

        bg, fg, axes_bg = self._theme()
        self.ax.clear()
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(axes_bg)
        cmap = plt.get_cmap("tab20", len(class_names))
        for i, name in enumerate(class_names):
            if visible_classes is not None and name not in visible_classes:
                continue
            mask = np.array(labels) == i
            if mask.sum() == 0:
                continue
            self.ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
                            s=12, alpha=0.7, color=cmap(i), label=name)
        self.ax.legend(fontsize=7, loc="best",
                       facecolor=axes_bg, labelcolor=fg,
                       markerscale=1.5)
        self.ax.set_title(title, color=fg, fontsize=11)
        self.ax.tick_params(colors=fg)
        self.ax.view_init(elev=self._DEFAULT_ELEV, azim=self._DEFAULT_AZIM)
        self.draw()


# ─────────────────────────────────────────────
# Main Tab Widget
# ─────────────────────────────────────────────
class DataVisualizationTab(QWidget):
    """
    Dimensionality-reduction visualisation tab.

    Left panel  : data-loading controls + algorithm hyper-parameters
    Right panel : 2-D scatter  |  3-D scatter  (side by side)
    """

    def __init__(self, dataset_manager=None, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self._thread      = None
        self._coords_2d   = None
        self._coords_3d   = None
        self._labels      = None
        self._class_names = []
        self._setup_ui()
        install_wheel_blocker(self)

    # ── UI construction ──────────────────────────────────────────────
    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_left_panel(), 1)
        root.addWidget(self._build_right_panel(), 3)

    def _build_left_panel(self):
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # ── Dataset path ─────────────────────────────────────────────
        layout.addWidget(self._section_label("Dataset"))

        self.dataset_path_label = QLabel("(not loaded)")
        self.dataset_path_label.setWordWrap(True)
        self.dataset_path_label.setProperty("class", "section-subtitle")
        layout.addWidget(self.dataset_path_label)

        load_btn = QPushButton("📂 Load waveform_data/")
        load_btn.setMinimumHeight(32)
        load_btn.clicked.connect(self._load_dataset)
        layout.addWidget(load_btn)

        self.load_registry_btn = QPushButton("📂 Load from Datasets Registry")
        self.load_registry_btn.setMinimumHeight(32)
        self.load_registry_btn.setEnabled(self.dataset_manager is not None)
        self.load_registry_btn.setToolTip("Load datasets directly from the shared datasets/ registry")
        self.load_registry_btn.clicked.connect(self._load_from_registry)
        layout.addWidget(self.load_registry_btn)

        # max samples per class
        row = QHBoxLayout()
        row.addWidget(QLabel("Max samples / class"))
        self.max_samples_spin = QSpinBox()
        self.max_samples_spin.setRange(10, 5000)
        self.max_samples_spin.setValue(200)
        row.addWidget(self.max_samples_spin)
        layout.addLayout(row)

        # ── Method selector ──────────────────────────────────────────
        layout.addSpacing(8)
        layout.addWidget(self._section_label("Method"))

        self.method_combo = QComboBox()
        self.method_combo.addItems(["UMAP", "t-SNE", "PCA"])
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        layout.addWidget(self.method_combo)

        # ── Hyper-parameter panels (stacked via show/hide) ────────────
        self.umap_group  = self._build_umap_params()
        self.tsne_group  = self._build_tsne_params()
        self.pca_group   = self._build_pca_params()

        layout.addWidget(self.umap_group)
        layout.addWidget(self.tsne_group)
        layout.addWidget(self.pca_group)

        self._on_method_changed("UMAP")   # show correct panel

        # ── Feature options ──────────────────────────────────────────
        feat_group = QGroupBox("Feature Extraction")
        feat_lay = QVBoxLayout(feat_group)
        self.feat_raw      = QCheckBox("Raw samples (first 256)")
        self.feat_fft      = QCheckBox("FFT magnitude")
        self.feat_stats    = QCheckBox("Statistical features")
        self.feat_raw.setChecked(True)
        self.feat_fft.setChecked(True)
        self.feat_stats.setChecked(False)
        for cb in [self.feat_raw, self.feat_fft, self.feat_stats]:
            feat_lay.addWidget(cb)
        layout.addWidget(feat_group)

        # -- Class filter (this allows to select signal classes)
        layout.addSpacing(8)
        self.class_filter_group = QGroupBox("Class Filter")
        self._class_filter_inner = QVBoxLayout(self.class_filter_group)
        self._class_checkboxes = {}   # class_name -> QCheckBox
        self._class_filter_placeholder = QLabel("Load data first")
        self._class_filter_placeholder.setProperty("class", "section-subtitle")
        self._class_filter_inner.addWidget(self._class_filter_placeholder)
        # Select All / None buttons
        cls_btn_row = QHBoxLayout()
        all_btn  = QPushButton("All")
        none_btn = QPushButton("None")
        all_btn.setFixedHeight(32)
        none_btn.setFixedHeight(32)
        all_btn.clicked.connect(lambda: self._set_all_classes(True))
        none_btn.clicked.connect(lambda: self._set_all_classes(False))
        cls_btn_row.addWidget(all_btn)
        cls_btn_row.addWidget(none_btn)
        self._class_filter_inner.addLayout(cls_btn_row)
        layout.addWidget(self.class_filter_group)

        # ── Run button ───────────────────────────────────────────────
        layout.addSpacing(8)
        self.run_btn = QPushButton("▶ Run Visualization")
        self.run_btn.setMinimumHeight(36)
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # indeterminate
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setProperty("class", "section-subtitle")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("card")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    # ── hyper-parameter group boxes ──────────────────────────────────
    def _build_umap_params(self):
        g = QGroupBox("UMAP Parameters")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("n_neighbors"), 0, 0)
        self.umap_neighbors = QSpinBox()
        self.umap_neighbors.setRange(2, 200)
        self.umap_neighbors.setValue(15)
        self.umap_neighbors.setToolTip(
            "Controls local vs global structure.\n"
            "Small = local detail | Large = global structure"
        )
        lay.addWidget(self.umap_neighbors, 0, 1)

        lay.addWidget(QLabel("min_dist"), 1, 0)
        self.umap_min_dist = QDoubleSpinBox()
        self.umap_min_dist.setRange(0.0, 1.0)
        self.umap_min_dist.setSingleStep(0.05)
        self.umap_min_dist.setValue(0.1)
        self.umap_min_dist.setToolTip(
            "Minimum distance between points in embedding.\n"
            "Small = tighter clusters | Large = spread out"
        )
        lay.addWidget(self.umap_min_dist, 1, 1)

        lay.addWidget(QLabel("metric"), 2, 0)
        self.umap_metric = QComboBox()
        self.umap_metric.addItems(["euclidean", "cosine", "manhattan", "correlation"])
        lay.addWidget(self.umap_metric, 2, 1)

        return g

    def _build_tsne_params(self):
        g = QGroupBox("t-SNE Parameters")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("perplexity"), 0, 0)
        self.tsne_perplexity = QSpinBox()
        self.tsne_perplexity.setRange(5, 500)
        self.tsne_perplexity.setValue(30)
        self.tsne_perplexity.setToolTip(
            "Roughly = expected cluster size.\n"
            "Typical range: 5–50"
        )
        lay.addWidget(self.tsne_perplexity, 0, 1)

        lay.addWidget(QLabel("learning_rate"), 1, 0)
        self.tsne_lr = QDoubleSpinBox()
        self.tsne_lr.setRange(10, 1000)
        self.tsne_lr.setValue(200)
        lay.addWidget(self.tsne_lr, 1, 1)

        lay.addWidget(QLabel("n_iter"), 2, 0)
        self.tsne_niter = QSpinBox()
        self.tsne_niter.setRange(250, 5000)
        self.tsne_niter.setValue(1000)
        lay.addWidget(self.tsne_niter, 2, 1)

        return g

    def _build_pca_params(self):
        g = QGroupBox("PCA Parameters")
        lay = QVBoxLayout(g)
        lay.addWidget(QLabel("No hyper-parameters required.\nPCA is deterministic."))
        return g

    def _build_right_panel(self):
        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 2D canvas
        left_box = QGroupBox("2D Projection")
        left_lay = QVBoxLayout(left_box)
        self.canvas_2d = ScatterCanvas2D()
        left_lay.addWidget(self.canvas_2d)
        layout.addWidget(left_box, 1)

        # 3D canvas
        right_box = QGroupBox("3D Projection")
        right_lay = QVBoxLayout(right_box)
        self.canvas_3d = ScatterCanvas3D()
        right_lay.addWidget(self.canvas_3d)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedHeight(28)
        reset_btn.setToolTip("Return to default elevation / azimuth")
        reset_btn.clicked.connect(self.canvas_3d.reset_view)
        right_lay.addWidget(reset_btn)

        layout.addWidget(right_box, 1)

        return panel

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _section_label(text):
        lbl = QLabel(text)
        lbl.setProperty("class", "section-title")
        return lbl

    def _on_method_changed(self, method):
        self.umap_group.setVisible(method == "UMAP")
        self.tsne_group.setVisible(method == "t-SNE")
        self.pca_group.setVisible(method == "PCA")

    # ── data loading ─────────────────────────────────────────────────
    def _find_waveform_data_dir(self):
        """Find waveform_data/ folder - check settings first, then local paths."""
        # First check if user set a data folder in settings
        settings = QSettings("MyCompany", "MixedSignalGUI")
        data_dir = settings.value("dataPath", "")
        
        if data_dir and os.path.isdir(data_dir):
            # Check if this is the waveform_data folder itself
            if os.path.basename(data_dir) == "waveform_data":
                return data_dir
            # Or if waveform_data is inside it
            wf_dir = os.path.join(data_dir, "waveform_data")
            if os.path.isdir(wf_dir):
                return wf_dir
            # Otherwise use the data_dir directly if it has subdirectories
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            if subdirs:
                return data_dir
        
        # Fall back to searching relative to this file
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "waveform_data"),
            os.path.join(os.path.dirname(here), "waveform_data"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        return None

    def _load_dataset(self):
        data_dir = self._find_waveform_data_dir()
        if data_dir is None:
            QMessageBox.warning(self, "Not found",
                                "Could not find waveform_data/ folder.")
            return
        self.dataset_path_label.setText(data_dir)
        self._data_dir = data_dir
        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.status_label.setText(
            f"Found {len(classes)} classes: {', '.join(classes)}"
        )

        self._class_filter_placeholder.setVisible(False)
        # remove prev . checkbox
        for cb in self._class_checkboxes.values():
            self._class_filter_inner.removeWidget(cb)
            cb.deleteLater()
        self._class_checkboxes = {}
        # generate checkbox per class
        insert_pos = 1
        for cls in classes:
            cb = QCheckBox(cls)
            cb.setChecked(True)
            cb.stateChanged.connect(self._apply_class_filter)
            self._class_filter_inner.insertWidget(insert_pos, cb)
            self._class_checkboxes[cls] = cb
            insert_pos += 1


    def _load_from_registry(self):
        """Load datasets from the DatasetManager registry and set up class checkboxes."""
        if self.dataset_manager is None:
            return

        entries = self.dataset_manager.scan()
        base_entries = [
            e for e in entries
            if not e.get('augmented', False) and e.get('modulation')
        ]
        if not base_entries:
            QMessageBox.warning(self, "No data", "No datasets with modulation metadata found. Generate some first.")
            return

        # Group by modulation — each unique modulation = one class
        from collections import defaultdict
        by_modulation = defaultdict(list)
        for entry in base_entries:
            mod = entry['modulation']
            npy = entry.get('_npy_path', '')
            if npy:
                by_modulation[mod].append(npy)

        # Store as a virtual "registry dir" by building a temp structure
        # that _collect_features_from_files can consume directly
        self._registry_data = dict(by_modulation)
        self._data_dir = None   # signal to _run to use registry path

        classes = sorted(by_modulation.keys())
        self.dataset_path_label.setText(
            f"Registry: {len(base_entries)} datasets, {len(classes)} classes"
        )
        self.status_label.setText(f"Found {len(classes)} classes: {', '.join(classes)}")

        # Rebuild class filter checkboxes
        self._class_filter_placeholder.setVisible(False)
        for cb in self._class_checkboxes.values():
            self._class_filter_inner.removeWidget(cb)
            cb.deleteLater()
        self._class_checkboxes = {}
        insert_pos = 1
        for cls in classes:
            cb = QCheckBox(cls)
            cb.setChecked(True)
            cb.stateChanged.connect(self._apply_class_filter)
            self._class_filter_inner.insertWidget(insert_pos, cb)
            self._class_checkboxes[cls] = cb
            insert_pos += 1

    def _extract_features(self, sig, use_raw, use_fft, use_stats):
        """Build one feature vector from a raw signal.

        Complex baseband is kept complex so both quadrature components
        contribute: raw samples become interleaved [I0, Q0, I1, Q1, ...] and
        the spectrum uses the full two-sided FFT.  Taking ``.real`` here (as
        this used to) threw Q away, so QAM and PSK collapsed onto each other
        in the projection — their in-phase components are near-identical and
        the phase information that separates them lives in Q.

        Real passband signals keep their previous raw and spectrum blocks
        byte-for-byte (they skip the interleave and still use ``rfft``).  The
        stats block does change for them: it now describes ``abs(sig)`` rather
        than the signed samples, because percentiles are undefined for complex
        values and one definition has to cover both.  ``max`` and the energy
        sum are unaffected; mean/std/percentiles shift, so a saved projection
        from before this change is not directly comparable.

        Either dtype yields the same feature width (256 raw + 256 spectrum +
        8 stats), so a folder mixing real and complex files still stacks into
        a single matrix.
        """
        sig = np.asarray(sig).flatten()
        is_complex = np.iscomplexobj(sig)
        feats = []

        if use_raw:
            if is_complex:
                # 128 complex samples -> 256 interleaved reals, so the raw
                # feature block stays 256 wide however the data arrives.
                head = sig[:128]
                raw = np.empty(2 * len(head), dtype=np.float32)
                raw[0::2] = head.real
                raw[1::2] = head.imag
            else:
                raw = sig[:256].astype(np.float32)
            if len(raw) < 256:
                raw = np.pad(raw, (0, 256 - len(raw)))
            feats.append(raw)

        if use_fft:
            # np.fft.rfft rejects complex input outright, so use the two-sided
            # transform.  For complex baseband the negative half is real
            # information (it is not a mirror of the positive half).
            window = sig[:512]
            if is_complex:
                spec = np.fft.fftshift(np.fft.fft(window))
            else:
                spec = np.fft.rfft(window)
            fft_mag = np.abs(spec)[:256].astype(np.float32)
            if len(fft_mag) < 256:
                fft_mag = np.pad(fft_mag, (0, 256 - len(fft_mag)))
            fft_mag /= (fft_mag.max() + 1e-9)
            feats.append(fft_mag)

        if use_stats:
            # Percentiles are undefined for complex values, so describe the
            # magnitude envelope and report I/Q spread separately.
            mag = np.abs(sig)
            stats = np.array([
                mag.mean(), mag.std(), np.percentile(mag, 25),
                np.percentile(mag, 75), mag.max(),
                float(np.sum(mag ** 2)),
                sig.real.std(), sig.imag.std() if is_complex else 0.0,
            ], dtype=np.float32)
            feats.append(stats)

        return np.concatenate(feats) if feats else None

    def _collect_features(self, data_dir, max_per_class):
        """Load .npy files and extract feature vectors."""
        classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        X_list, y_list = [], []

        use_raw   = self.feat_raw.isChecked()
        use_fft   = self.feat_fft.isChecked()
        use_stats = self.feat_stats.isChecked()

        for ci, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
            files = files[:max_per_class]
            for fpath in files:
                sig = np.load(os.path.join(cls_dir, fpath))
                feats = self._extract_features(sig, use_raw, use_fft, use_stats)
                if feats is not None:
                    X_list.append(feats)
                    y_list.append(ci)

        if not X_list:
            raise ValueError("No data found! Generate waveforms first.")

        X = np.vstack(X_list).astype(np.float32)
        # Normalize
        X -= X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        X /= std
        return X, y_list, classes

    def _collect_features_from_registry(self, max_per_class):
        """Like _collect_features but reads from self._registry_data (npy file lists)."""
        registry = getattr(self, '_registry_data', {})
        if not registry:
            raise ValueError("No registry data loaded.")

        classes = sorted(registry.keys())
        X_list, y_list = [], []

        use_raw   = self.feat_raw.isChecked()
        use_fft   = self.feat_fft.isChecked()
        use_stats = self.feat_stats.isChecked()

        for ci, cls in enumerate(classes):
            files = registry[cls][:max_per_class]
            for fpath in files:
                try:
                    sig = np.load(fpath)
                    feats = self._extract_features(sig, use_raw, use_fft, use_stats)
                    if feats is not None:
                        X_list.append(feats)
                        y_list.append(ci)
                except Exception as e:
                    print(f"[DataViz] Could not load {fpath}: {e}")

        if not X_list:
            raise ValueError("No data could be loaded from the registry.")

        X = np.vstack(X_list).astype(np.float32)
        X -= X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1
        X /= std
        return X, y_list, classes

    # ── run ──────────────────────────────────────────────────────────
    def _run(self):
        if not hasattr(self, "_data_dir") or (self._data_dir is None and not hasattr(self, "_registry_data")):
            # auto-detect waveform_data/
            data_dir = self._find_waveform_data_dir()
            if data_dir is None:
                QMessageBox.warning(self, "No data",
                                    "Load a dataset folder or use 'Load from Datasets Registry' first.")
                return
            self._data_dir = data_dir

        try:
            self.status_label.setText("Extracting features…")
            if self._data_dir is not None:
                X, y, classes = self._collect_features(
                    self._data_dir, self.max_samples_spin.value()
                )
            else:
                X, y, classes = self._collect_features_from_registry(
                    self.max_samples_spin.value()
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self._class_names = classes
        method = self.method_combo.currentText()

        params = {}
        if method == "UMAP":
            params = {
                "n_neighbors": self.umap_neighbors.value(),
                "min_dist":    self.umap_min_dist.value(),
                "metric":      self.umap_metric.currentText(),
            }
        elif method == "t-SNE":
            params = {
                "perplexity":    self.tsne_perplexity.value(),
                "learning_rate": self.tsne_lr.value(),
                "n_iter":        self.tsne_niter.value(),
            }

        self.run_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText(f"Running {method}…  (may take ~30 s)")

        self._thread = DimReduceThread(X, y, method, params)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.progress.connect(lambda m: self.status_label.setText(m))
        self._thread.start()

    # below are for setting signal clasess (checkbox)
    def _set_all_classes(self, checked: bool):
        for cb in self._class_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._apply_class_filter()

    def _get_selected_classes(self):
        """Return set of class names currently checked."""
        return {name for name, cb in self._class_checkboxes.items() if cb.isChecked()}

    def _apply_class_filter(self):
        """Re-draw both plots using only selected classes (no re-computation)."""
        if not hasattr(self, "_coords_2d") or self._coords_2d is None:
            return
        selected = self._get_selected_classes()
        if not selected:
            return
        method = self.method_combo.currentText()
        self.canvas_2d.plot(self._coords_2d, self._labels, self._class_names,
                            title=f"{method} — 2D",
                            visible_classes=selected)
        self.canvas_3d.plot(self._coords_3d, self._labels, self._class_names,
                            title=f"{method} — 3D",
                            visible_classes=selected)

    # def _on_finished(self, coords_2d, coords_3d, labels):
    #     self.progress_bar.setVisible(False)
    #     self.run_btn.setEnabled(True)
    #     method = self.method_combo.currentText()
    #     self.status_label.setText(
    #         f"Done! {len(labels)} samples · {len(self._class_names)} classes"
    #     )
    #     self.canvas_2d.plot(coords_2d, labels, self._class_names,
    #                         title=f"{method} — 2D")
    #     self.canvas_3d.plot(coords_3d, labels, self._class_names,
    #                         title=f"{method} — 3D")
    #     if self._thread:
    #         self._thread.quit()
    #         self._thread.wait()
    #         self._thread = None

    def _on_finished(self, coords_2d, coords_3d, labels):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        method = self.method_combo.currentText()

        #checkbox coordinate 
        self._coords_2d = coords_2d
        self._coords_3d = coords_3d
        self._labels    = labels

        self.status_label.setText(
            f"Done! {len(labels)} samples · {len(self._class_names)} classes"
        )
        selected = self._get_selected_classes() or set(self._class_names)
        self.canvas_2d.plot(coords_2d, labels, self._class_names,
                            title=f"{method} — 2D",
                            visible_classes=selected)
        self.canvas_3d.plot(coords_3d, labels, self._class_names,
                            title=f"{method} — 3D",
                            visible_classes=selected)
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.run_btn.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Visualization Error", msg)
