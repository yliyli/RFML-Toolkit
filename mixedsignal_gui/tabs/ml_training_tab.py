from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QGridLayout, QFrame, QProgressBar, QFileDialog,
                               QListWidget, QListWidgetItem, QDoubleSpinBox, QScrollArea,
                               QCheckBox, QMessageBox, QToolButton,
                               QSpinBox, QLineEdit)
from PySide6.QtCore import Qt, Signal, QSettings

import os
import numpy as np
import torch

from mixedsignal_gui.widgets.training_chart import TrainingChartWidget
from mixedsignal_gui.widgets.wheel_filter import install_wheel_blocker


class MLTrainingTab(QWidget):
    """ML Training configuration and visualization tab"""
    # Emitted when a model finishes training: (model_path, class_labels_list)
    trained_model_ready = Signal(str, list)

    def __init__(self, dataset_manager=None):
        super().__init__()
        self.dataset_manager = dataset_manager
        self.external_model_plugins = {}

        self.setup_ui()
        install_wheel_blocker(self)
        if self.dataset_manager is not None:
            self.load_registry_btn.setEnabled(True)

    def setup_ui(self):
        """Initialize the UI components"""
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(0, 0, 0, 0)

        # Left panel - Training Configuration
        left_panel = self.create_configuration_panel()
        layout.addWidget(left_panel, 2)

        # Right panel - Charts
        right_panel = self.create_charts_panel()
        layout.addWidget(right_panel, 3)

    def create_configuration_panel(self):
        """Create the training configuration panel (scrollable)"""
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # ── Title ──────────────────────────────────────────────────────────
        title = QLabel("\u2699\ufe0f Training Configuration")
        title.setProperty("class", "section-title")
        subtitle = QLabel("Configure ML model parameters")
        subtitle.setProperty("class", "section-subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        # ── Dataset buttons — split into two rows so they don't clip ──────
        data_row1 = QHBoxLayout()
        self.add_data_btn = QPushButton("Add Data Folder")
        self.add_data_btn.clicked.connect(self.add_data_folder)
        data_row1.addWidget(self.add_data_btn)

        self.load_registry_btn = QPushButton("\U0001f4c2 Load from Datasets")
        self.load_registry_btn.setToolTip("Group datasets by modulation type and load as training classes")
        self.load_registry_btn.clicked.connect(self.load_from_registry)
        self.load_registry_btn.setEnabled(False)
        data_row1.addWidget(self.load_registry_btn)

        self.quick_load_btn = QPushButton("\U0001f4e6 Quick Load Dataset")
        self.quick_load_btn.setToolTip("Auto-load class folders from waveform_data/ directory")
        self.quick_load_btn.clicked.connect(self.quick_load_dataset)
        data_row1.addWidget(self.quick_load_btn)
        layout.addLayout(data_row1)

        data_row2 = QHBoxLayout()
        self.remove_data_btn = QPushButton("Remove Selected")
        self.remove_data_btn.clicked.connect(self.remove_selected_dataset)
        self.remove_data_btn.setEnabled(False)
        data_row2.addWidget(self.remove_data_btn)

        self.clear_data_btn = QPushButton("Clear All")
        self.clear_data_btn.clicked.connect(self.clear_datasets)
        self.clear_data_btn.setEnabled(False)
        data_row2.addWidget(self.clear_data_btn)
        data_row2.addStretch()
        layout.addLayout(data_row2)

        # ── Dataset list ───────────────────────────────────────────────────
        self.dataset_list = QListWidget()
        self.dataset_list.setMinimumHeight(80)
        self.dataset_list.setMaximumHeight(160)
        self.dataset_list.itemSelectionChanged.connect(self.on_dataset_selection_changed)
        layout.addWidget(self.dataset_list)

        # ── Model architecture ─────────────────────────────────────────────
        arch_label = QLabel("Model Architecture")
        arch_label.setProperty("class", "stat-label")
        layout.addWidget(arch_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["SimpleCNN", "TinyConv", "MLP", "ResNet1DOptimized"])
        arch_row = QHBoxLayout()
        arch_row.addWidget(self.model_combo, 1)
        self.load_model_plugin_btn = QPushButton("Load Python…")
        self.load_model_plugin_btn.clicked.connect(self._choose_model_plugin)
        arch_row.addWidget(self.load_model_plugin_btn)
        self.model_plugin_help_btn = QToolButton()
        self.model_plugin_help_btn.setText("?")
        self.model_plugin_help_btn.setToolTip("Custom PyTorch model requirements")
        self.model_plugin_help_btn.clicked.connect(self._show_model_plugin_help)
        arch_row.addWidget(self.model_plugin_help_btn)
        layout.addLayout(arch_row)

        # ── Model save path ────────────────────────────────────────────────
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save model to:"))
        self.model_save_path_edit = QLineEdit()
        self.model_save_path_edit.setPlaceholderText("Default: models/ folder")
        self.model_save_path_edit.setReadOnly(True)
        save_row.addWidget(self.model_save_path_edit, 1)
        browse_save_btn = QPushButton("Browse\u2026")
        browse_save_btn.clicked.connect(self._browse_save_path)
        save_row.addWidget(browse_save_btn)
        layout.addLayout(save_row)

        # ── Epochs & batch size ────────────────────────────────────────────
        eb_layout = QGridLayout()
        eb_layout.setColumnStretch(1, 1)
        eb_layout.setColumnStretch(3, 1)

        eb_layout.addWidget(QLabel("Epochs"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setMinimumWidth(80)
        eb_layout.addWidget(self.epochs_spin, 0, 1)

        eb_layout.addWidget(QLabel("Batch Size"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(64)
        self.batch_spin.setMinimumWidth(80)
        eb_layout.addWidget(self.batch_spin, 0, 3)

        layout.addLayout(eb_layout)

        # store selected datasets: label -> list[filepaths]
        self.datasets = {}
        # trainer reference
        self._trainer = None

        # ── Training Hyperparameters ──────────────────────────────────────
        training_card, self.training_hparams_toggle, th_layout = self._create_hparams_section(
            "Training Hyperparameters"
        )

        th_layout.addWidget(QLabel("Learning Rate"), 0, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.001)
        th_layout.addWidget(self.lr_spin, 0, 1)

        th_layout.addWidget(QLabel("Weight Decay"), 1, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.weight_decay_spin.setValue(1e-4)
        th_layout.addWidget(self.weight_decay_spin, 1, 1)

        th_layout.addWidget(QLabel("Label Smoothing"), 2, 0)
        self.label_smoothing_spin = QDoubleSpinBox()
        self.label_smoothing_spin.setRange(0.0, 1.0)
        self.label_smoothing_spin.setDecimals(2)
        self.label_smoothing_spin.setSingleStep(0.05)
        self.label_smoothing_spin.setValue(0.1)
        th_layout.addWidget(self.label_smoothing_spin, 2, 1)

        th_layout.addWidget(QLabel("Validation Split"), 3, 0)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.5)
        self.val_split_spin.setDecimals(2)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.2)
        th_layout.addWidget(self.val_split_spin, 3, 1)

        th_layout.addWidget(QLabel("Gradient Clip"), 4, 0)
        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.0, 100.0)
        self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setSingleStep(0.1)
        self.grad_clip_spin.setValue(1.0)
        th_layout.addWidget(self.grad_clip_spin, 4, 1)

        self.training_hparams_toggle.toggled.connect(th_layout.parentWidget().setVisible)
        layout.addWidget(training_card)

        # ── Per-Model Hyperparameters ─────────────────────────────────────
        model_card, self.model_hparams_toggle, self._mh_layout = self._create_hparams_section(
            "Model Hyperparameters"
        )

        self._mh_base_filters_label = QLabel("Base Filters")
        self._mh_layout.addWidget(self._mh_base_filters_label, 0, 0)
        self.resnet_base_filters_spin = QDoubleSpinBox()
        self.resnet_base_filters_spin.setRange(8, 512)
        self.resnet_base_filters_spin.setDecimals(0)
        self.resnet_base_filters_spin.setSingleStep(8)
        self.resnet_base_filters_spin.setValue(64)
        self._mh_layout.addWidget(self.resnet_base_filters_spin, 0, 1)

        self._mh_dropout_label = QLabel("Dropout")
        self._mh_layout.addWidget(self._mh_dropout_label, 1, 0)
        self.resnet_dropout_spin = QDoubleSpinBox()
        self.resnet_dropout_spin.setRange(0.0, 0.9)
        self.resnet_dropout_spin.setDecimals(2)
        self.resnet_dropout_spin.setSingleStep(0.05)
        self.resnet_dropout_spin.setValue(0.2)
        self._mh_layout.addWidget(self.resnet_dropout_spin, 1, 1)

        self._mh_no_params_label = QLabel("No editable hyperparameters for this model.")
        self._mh_layout.addWidget(self._mh_no_params_label, 0, 0, 1, 2)

        self.model_hparams_toggle.toggled.connect(self._update_model_hparams_visibility)
        self.model_combo.currentTextChanged.connect(self._update_model_hparams_visibility)
        layout.addWidget(model_card)

        # ── Training summary stats (inside a card frame) ───────────────────
        stats_frame = QFrame()
        stats_frame.setObjectName("card")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setSpacing(4)
        stats_layout.setContentsMargins(12, 8, 12, 8)

        self.val_labels = {
            "Training Samples":   QLabel("--"),
            "Validation Samples": QLabel("--"),
            "Batch Size":         QLabel(str(self.batch_spin.value())),
            "Epochs":             QLabel(str(self.epochs_spin.value())),
        }

        for i, (label_text, value_widget) in enumerate(self.val_labels.items()):
            lbl = QLabel(label_text)
            lbl.setProperty("class", "stat-label")
            value_widget.setProperty("class", "stat-value")
            value_widget.setAlignment(Qt.AlignRight)
            stats_layout.addWidget(lbl, i, 0)
            stats_layout.addWidget(value_widget, i, 1)

        self.batch_spin.valueChanged.connect(lambda v: self.val_labels["Batch Size"].setText(str(v)))
        self.epochs_spin.valueChanged.connect(lambda v: self.val_labels["Epochs"].setText(str(v)))
        layout.addWidget(stats_frame)

        # ── Progress section ───────────────────────────────────────────────
        progress_header = QHBoxLayout()
        progress_label = QLabel("Training Progress")
        progress_label.setProperty("class", "stat-label")
        self.progress_value = QLabel("Epoch 0/0")
        self.progress_value.setProperty("class", "stat-value")
        self.progress_value.setAlignment(Qt.AlignRight)
        progress_header.addWidget(progress_label)
        progress_header.addWidget(self.progress_value)
        layout.addLayout(progress_header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # Push the button to the bottom
        layout.addStretch()

        # ── Start Training button ──────────────────────────────────────────
        self.train_btn = QPushButton("\u25b6  Start Training")
        self.train_btn.setObjectName("primaryButton")
        self.train_btn.setMinimumHeight(38)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        layout.addWidget(self.train_btn)

        # ── Status label ───────────────────────────────────────────────────
        self.status_label = QLabel("Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setProperty("class", "stat-label")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Wrap in scroll area
        scroll = QScrollArea()
        scroll.setObjectName("card")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        return scroll

    # ── Section helpers ───────────────────────────────────────────────────

    def _create_hparams_section(self, title):
        """Create a card-style section with a checkbox header and hidden content area."""
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 14)
        card_layout.setSpacing(10)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        toggle = QCheckBox()
        toggle.setChecked(False)
        header.addWidget(toggle)

        title_label = QLabel(title)
        title_label.setProperty("class", "section-title")
        header.addWidget(title_label)
        header.addStretch()

        card_layout.addLayout(header)

        content = QWidget()
        content_layout = QGridLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setHorizontalSpacing(12)
        content_layout.setVerticalSpacing(12)
        content.setVisible(False)
        card_layout.addWidget(content)

        toggle.toggled.connect(content.setVisible)

        return card, toggle, content_layout

    def _update_model_hparams_visibility(self, *_):
        """Show/hide model-specific hyperparameter widgets based on selected model."""
        if not self.model_hparams_toggle.isChecked():
            return
        is_resnet = self.model_combo.currentText() == 'ResNet1DOptimized'
        self._mh_base_filters_label.setVisible(is_resnet)
        self.resnet_base_filters_spin.setVisible(is_resnet)
        self._mh_dropout_label.setVisible(is_resnet)
        self.resnet_dropout_spin.setVisible(is_resnet)
        self._mh_no_params_label.setVisible(not is_resnet)

    # ── Dataset management ─────────────────────────────────────────────────

    def add_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder", os.path.expanduser("~"))
        if not folder:
            return

        subdirs = [d for d in sorted(os.listdir(folder))
                   if os.path.isdir(os.path.join(folder, d))]
        subdir_with_files = []
        for d in subdirs:
            files = self._gather_dataset_files(os.path.join(folder, d))
            if files:
                subdir_with_files.append((d, files))

        if subdir_with_files:
            for sub_label, files in subdir_with_files:
                label = sub_label
                orig_label = label
                i = 1
                while label in self.datasets:
                    label = f"{orig_label}_{i}"
                    i += 1
                self.datasets[label] = files
                item = QListWidgetItem(f"{label} ({len(files)} files)")
                item.setData(Qt.UserRole, label)
                self.dataset_list.addItem(item)
            self.clear_data_btn.setEnabled(True)
            self.status_label.setText(f"Loaded {len(subdir_with_files)} classes from {os.path.basename(folder)}")
            self._update_train_button_state()
            return

        label = os.path.basename(folder.rstrip(os.sep)) or folder
        orig_label = label
        i = 1
        while label in self.datasets:
            label = f"{orig_label}_{i}"
            i += 1

        files = self._gather_dataset_files(folder)
        if not files:
            self.status_label.setText("No supported files (.npy/.npz/.csv) found")
            return

        self.datasets[label] = files
        item = QListWidgetItem(f"{label} ({len(files)} files)")
        item.setData(Qt.UserRole, label)
        self.dataset_list.addItem(item)
        self.clear_data_btn.setEnabled(True)
        self._update_train_button_state()

    def remove_selected_dataset(self):
        items = self.dataset_list.selectedItems()
        if not items:
            return
        for it in items:
            label = it.data(Qt.UserRole)
            if label in self.datasets:
                del self.datasets[label]
            row = self.dataset_list.row(it)
            self.dataset_list.takeItem(row)
        self.remove_data_btn.setEnabled(False)
        if self.dataset_list.count() == 0:
            self.clear_data_btn.setEnabled(False)
        self._update_train_button_state()

    def clear_datasets(self):
        self.datasets.clear()
        self.dataset_list.clear()
        self.remove_data_btn.setEnabled(False)
        self.clear_data_btn.setEnabled(False)
        self._update_train_button_state()

    def on_dataset_selection_changed(self):
        self.remove_data_btn.setEnabled(len(self.dataset_list.selectedItems()) > 0)

    def _update_train_button_state(self):
        valid_classes = [k for k, v in self.datasets.items() if v]
        can_train = len(valid_classes) >= 2
        self.train_btn.setEnabled(can_train)
        if valid_classes and not can_train:
            self.status_label.setText(
                f"Need at least 2 classes to train (currently {len(valid_classes)}). "
                "Add more data folders or use Batch Generate on the Waveform tab."
            )

    def _gather_dataset_files(self, folder):
        """Return list of candidate data files in folder (.npy, .npz, .csv)."""
        exts = ('.npy', '.npz', '.csv')
        files = []
        try:
            for entry in os.listdir(folder):
                path = os.path.join(folder, entry)
                if os.path.isfile(path) and entry.lower().endswith(exts):
                    files.append(path)
        except Exception as e:
            print(f"Error listing dataset folder: {e}")
        return sorted(files)

    def _load_sample_from_file(self, path):
        """Load a sample from file. Supports .npy/.npz/.csv."""
        try:
            if path.lower().endswith('.npy'):
                data = np.load(path, allow_pickle=True)
                return data
            if path.lower().endswith('.npz'):
                data = np.load(path, allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    keys = list(data.keys())
                    return data[keys[0]] if keys else None
            if path.lower().endswith('.csv'):
                data = np.loadtxt(path, delimiter=',')
                return data
        except Exception as e:
            print(f"Failed to load sample {path}: {e}")
        return None

    # ── Charts panel ───────────────────────────────────────────────────────

    def create_charts_panel(self):
        """Create the charts visualization panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        loss_card = self.create_chart_card(
            "Training & Validation Loss",
            "Loss convergence over epochs",
            "loss"
        )
        layout.addWidget(loss_card)

        acc_card = self.create_chart_card(
            "Training & Validation Accuracy",
            "Classification accuracy over epochs",
            "accuracy"
        )
        layout.addWidget(acc_card)

        return panel

    def create_chart_card(self, title, subtitle, chart_type):
        """Create a chart card with title and legend"""
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title_label = QLabel(title)
        title_label.setProperty("class", "card-title")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setProperty("class", "section-subtitle")
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)

        if chart_type == "loss":
            self.loss_chart = TrainingChartWidget("Loss", "Loss", cap_at_one=False)
            layout.addWidget(self.loss_chart)
            legend = self.create_legend(
                [("\u2192 Training Loss", "#3b82f6"), ("\u2192 Validation Loss", "#fb923c")]
            )
        else:
            self.acc_chart = TrainingChartWidget("Accuracy", "Accuracy (%)", cap_at_one=True)
            layout.addWidget(self.acc_chart)
            legend = self.create_legend(
                [("\u2192 Training Accuracy", "#3b82f6"), ("\u2192 Validation Accuracy", "#fb923c")]
            )

        layout.addLayout(legend)
        return card

    def create_legend(self, items):
        """Create a legend layout for the chart"""
        legend_layout = QHBoxLayout()
        legend_layout.addStretch()
        for i, (text, color) in enumerate(items):
            label = QLabel(text)
            label.setStyleSheet(f"color: {color}; font-size: 12px;")
            legend_layout.addWidget(label)
            if i < len(items) - 1:
                legend_layout.addSpacing(20)
        legend_layout.addStretch()
        return legend_layout

    # ── Registry / quick-load ──────────────────────────────────────────────

    def load_from_registry(self):
        """Load datasets from the shared DatasetManager, grouped by modulation as classes."""
        if self.dataset_manager is None:
            return

        entries = self.dataset_manager.scan()
        # Any entry carrying a modulation label is usable, augmented or not.
        # This used to require `not augmented`, which made the whole channel
        # workflow untrainable: a bulk augmentation run writes 900 files that all
        # carry augmented=True, so the registry reported "no datasets" for exactly
        # the folder the user had just produced.  Augmented examples are the point
        # of the channel tab -- they keep the modulation field of the waveform they
        # came from, which is the class label.
        base_entries = [e for e in entries if e.get('modulation')]

        # Batch Generate records data_split on every sample it writes, but
        # nothing ever read it: the held-out test samples were loaded straight
        # into training, so a 75/25 split had no effect and every accuracy the
        # app reported was measured on data the model had trained on.  Entries
        # with no data_split (older datasets, Quick Test Data) are kept, since
        # they were never part of a split in the first place.
        held_out = [e for e in base_entries if e.get('data_split') == 'test']
        base_entries = [e for e in base_entries if e.get('data_split') != 'test']

        if not base_entries:
            self.status_label.setText("No datasets with modulation metadata found \u2013 generate some first.")
            return

        from collections import defaultdict
        by_modulation = defaultdict(list)
        for entry in base_entries:
            mod = entry['modulation']
            npy = entry.get('_npy_path', '')
            if npy:
                by_modulation[mod].append(npy)

        if len(by_modulation) < 2:
            self.status_label.setText(
                f"Only {len(by_modulation)} modulation class(es) found \u2013 need \u2265 2. "
                "Use Batch Generate on the Waveform tab."
            )
            return

        added = 0
        for mod, files in sorted(by_modulation.items()):
            label = mod
            base_label = label
            i = 1
            while label in self.datasets:
                label = f"{base_label}_{i}"
                i += 1
            self.datasets[label] = files
            item = QListWidgetItem(f"{label} ({len(files)} files) [registry]")
            item.setData(Qt.UserRole, label)
            self.dataset_list.addItem(item)
            added += 1

        self.clear_data_btn.setEnabled(True)
        held_note = (f"; held back {len(held_out)} test sample(s) for the "
                     f"Inference tab" if held_out else "")
        self.status_label.setText(f"Loaded {added} class(es) from registry{held_note}")
        self._update_train_button_state()

    def _browse_save_path(self):
        """Let the user pick a directory to save the trained model into."""
        folder = QFileDialog.getExistingDirectory(self, "Select Model Save Folder")
        if folder:
            self.model_save_path_edit.setText(folder)

    def quick_load_dataset(self):
        """Auto-load class folders from waveform_data/ directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(script_dir)
        waveform_dir = os.path.join(base_dir, 'waveform_data')

        if not os.path.isdir(waveform_dir):
            waveform_dir = os.path.join(os.getcwd(), 'waveform_data')

        if not os.path.isdir(waveform_dir):
            self.status_label.setText("No waveform_data/ directory found")
            return

        self.clear_datasets()
        loaded = 0
        for entry in sorted(os.listdir(waveform_dir)):
            class_dir = os.path.join(waveform_dir, entry)
            if not os.path.isdir(class_dir):
                continue
            files = self._gather_dataset_files(class_dir)
            if not files:
                continue
            label = entry
            self.datasets[label] = files
            item = QListWidgetItem(f"{label} ({len(files)} files)")
            item.setData(Qt.UserRole, label)
            self.dataset_list.addItem(item)
            loaded += 1

        if loaded > 0:
            self.clear_data_btn.setEnabled(True)
            self.status_label.setText(f"Loaded {loaded} classes from waveform_data/")
        else:
            self.status_label.setText(
                f"No class subfolders with data found in {waveform_dir}. "
                "Use Batch Generate on the Waveform tab first."
            )
        self._update_train_button_state()

    # ── Training ───────────────────────────────────────────────────────────

    def _show_model_plugin_help(self):
        QMessageBox.information(
            self, "Custom PyTorch model",
            "Select a trusted Python file that defines\n\n"
            "build_model(num_classes, input_size, in_channels, **kwargs)\n\n"
            "The function must return a torch.nn.Module whose forward method "
            "returns unnormalized class logits. The model receives tensors shaped "
            "(batch, in_channels, input_size).")

    def _choose_model_plugin(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PyTorch model", "", "Python files (*.py)")
        if not path:
            return
        try:
            from mixedsignal_gui.backend.torch_models import inspect_external_model
            info = inspect_external_model(path)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid model plugin", str(exc))
            return
        display_name = f"Custom: {info["name"]}"
        if display_name not in self.external_model_plugins:
            self.model_combo.addItem(display_name)
        self.external_model_plugins[display_name] = info
        self.model_combo.setCurrentText(display_name)
        self.status_label.setText(f"Loaded custom model: {info["name"]}")

    def start_training(self):
        """Handle training start button click"""
        try:
            from mixedsignal_gui.backend.trainer import TrainerThread
        except ImportError as e:
            self.status_label.setText(f"PyTorch not installed: {e}")
            print(f"ERROR: Could not import TrainerThread: {e}")
            return

        if not self.datasets or len(self.datasets) < 2:
            self.status_label.setText("Add at least two data folders (classes) to train")
            return

        labels = list(self.datasets.keys())
        file_label_pairs = []
        for idx, label in enumerate(labels):
            files = self.datasets.get(label, [])
            for f in files:
                file_label_pairs.append((f, idx))

        if not file_label_pairs:
            self.status_label.setText("No supported files found in datasets")
            return

        lr = self.lr_spin.value()
        val_split = self.val_split_spin.value()
        weight_decay = self.weight_decay_spin.value()
        label_smoothing = self.label_smoothing_spin.value()
        grad_clip = self.grad_clip_spin.value()

        total_files = len(file_label_pairs)
        train_count = int(total_files * (1.0 - val_split))
        val_count = total_files - train_count

        self.val_labels["Training Samples"].setText(f"{train_count:,}")
        self.val_labels["Validation Samples"].setText(f"{val_count:,}")

        model_name = self.model_combo.currentText()
        plugin = self.external_model_plugins.get(model_name)
        epochs = int(self.epochs_spin.value())
        batch_size = int(self.batch_spin.value())

        model_hparams = {}
        if model_name == 'ResNet1DOptimized':
            model_hparams['base_filters'] = int(self.resnet_base_filters_spin.value())
            model_hparams['dropout'] = self.resnet_dropout_spin.value()

        # Reset progress UI
        self.progress_bar.setMaximum(epochs)
        self.progress_bar.setValue(0)
        self.progress_value.setText(f"Epoch 0/{epochs}")
        self.loss_chart.clear_data()
        self.acc_chart.clear_data()

        # Resolve save directory
        save_dir = self.model_save_path_edit.text().strip()
        if not save_dir:
            save_dir = QSettings("MyCompany", "MixedSignalGUI").value("modelPath", "")
        if not save_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(os.path.dirname(script_dir), "models")
        os.makedirs(save_dir, exist_ok=True)

        settings = QSettings("MyCompany", "MixedSignalGUI")
        compute_mode = settings.value("mode", "CPU")
        device = 'cuda' if (compute_mode == "GPU" and torch.cuda.is_available()) else 'cpu'

        self._trainer = TrainerThread(
            file_label_pairs, labels,
            model_name=model_name, epochs=epochs, batch_size=batch_size,
            lr=lr, val_split=val_split,
            weight_decay=weight_decay, label_smoothing=label_smoothing,
            grad_clip=grad_clip, model_hparams=model_hparams,
            model_plugin_path=plugin["path"] if plugin else None,
            save_dir=save_dir,
            device=device,
        )
        self._trainer.progress.connect(self.update_training_progress)
        self._trainer.finished.connect(self.on_training_finished)

        self._training_labels = labels

        self.status_label.setText("Training\u2026")
        self.train_btn.setEnabled(False)
        self.train_btn.setText("\u23f8  Training\u2026")
        self._trainer.start()
        print(f"Trainer started: model={model_name}, epochs={epochs}, batch_size={batch_size}, lr={lr}")

    def update_training_progress(self, epoch, total_epochs, train_loss, val_loss, train_acc, val_acc):
        """Update the training progress and labels with dynamic values"""
        self.progress_bar.setValue(epoch)
        self.progress_value.setText(f"Epoch {epoch}/{total_epochs}")

        self.loss_chart.add_data_point(train_loss, val_loss)
        self.acc_chart.add_data_point(train_acc, val_acc)

        self.val_labels["Epochs"].setText(f"{epoch} / {total_epochs}")

        self.status_label.setText(
            f"Epoch {epoch}: Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"Acc {train_acc * 100:.1f}% | Val Acc {val_acc * 100:.1f}%"
        )

    def on_training_finished(self, model_path):
        """Called when the trainer thread emits finished."""
        if model_path:
            self.status_label.setText(f"\u2705 Complete \u2013 saved: {os.path.basename(model_path)}")
            print(f"Model saved to: {model_path}")
            labels = getattr(self, '_training_labels', [])
            self.trained_model_ready.emit(model_path, labels)
        else:
            # An empty path means the run produced nothing worth saving.  Say
            # why: this used to be reported as "Complete - saved" with an
            # untrained model on disk.
            reason = getattr(self._trainer, "error", None)
            if reason:
                self.status_label.setText(f"❌ Training failed – no model saved")
                QMessageBox.critical(
                    self, "Training failed",
                    f"No model was saved because training did not complete:\n\n{reason}")
            else:
                self.status_label.setText("Training finished (no model saved)")
        self.train_btn.setEnabled(True)
        self.train_btn.setText("\u25b6  Start Training")
        if self._trainer:
            self._trainer.quit()
            self._trainer.wait()
            self._trainer = None
