"""
styles/stylesheet.py  –  Theming & Settings Dialog
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox,
    QFormLayout, QSlider, QFileDialog, QLineEdit, QFrame,
    QTabWidget, QWidget, QFontComboBox,
)
from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QFont, QPalette, QColor


# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------

THEMES = {
    "Light (Default)": {
        "bg_main":          "#f5f5f5",
        "bg_card":          "#ffffff",
        "bg_input":         "#ffffff",
        "bg_tab_active":    "#111827",
        "bg_tab_inactive":  "#ffffff",
        "accent":           "#111827",
        "accent_hover":     "#1f2937",
        "text_primary":     "#111827",
        "text_secondary":   "#6b7280",
        "text_muted":       "#9ca3af",
        "border":           "#e5e7eb",
        "border_focus":     "#111827",
        "progress_bg":      "#e5e7eb",
        "progress_fill":    "#111827",
        "selection_bg":     "#111827",
        "selection_fg":     "#ffffff",
        "info_bg":          "#eff6ff",
        "info_border":      "#bfdbfe",
        "tab_text_active":  "#ffffff",
        "tab_text_inactive":"#374151",
    },
    "Dark": {
        "bg_main":          "#1c1c1c",
        "bg_card":          "#272727",
        "bg_input":         "#2f2f2f",
        "bg_tab_active":    "#3a3a3a",
        "bg_tab_inactive":  "#272727",
        "accent":           "#d0d0d0",
        "accent_hover":     "#b0b0b0",
        "text_primary":     "#e8e8e8",
        "text_secondary":   "#9a9a9a",
        "text_muted":       "#666666",
        "border":           "#3a3a3a",
        "border_focus":     "#888888",
        "progress_bg":      "#3a3a3a",
        "progress_fill":    "#c0c0c0",
        "selection_bg":     "#555555",
        "selection_fg":     "#ffffff",
        "info_bg":          "#2f2f2f",
        "info_border":      "#3a3a3a",
        "tab_text_active":  "#ffffff",
        "tab_text_inactive":"#9a9a9a",
    },
    "Sunset": {
        "bg_main":          "#1a1a2e",
        "bg_card":          "#16213e",
        "bg_input":         "#0f3460",
        "bg_tab_active":    "#e94560",
        "bg_tab_inactive":  "#0f3460",
        "accent":           "#e94560",
        "accent_hover":     "#c73652",
        "text_primary":     "#eaeaea",
        "text_secondary":   "#a0a0b0",
        "text_muted":       "#6b6b80",
        "border":           "#2a2a4a",
        "border_focus":     "#e94560",
        "progress_bg":      "#2a2a4a",
        "progress_fill":    "#e94560",
        "selection_bg":     "#e94560",
        "selection_fg":     "#1a1a2e",
        "info_bg":          "#0f3460",
        "info_border":      "#2a2a4a",
        "tab_text_active":  "#ffffff",
        "tab_text_inactive":"#a0a0b0",
    },
    "Midnight Blue": {
        "bg_main":          "#0d1b2a",
        "bg_card":          "#1b2838",
        "bg_input":         "#1e3a5f",
        "bg_tab_active":    "#4fc3f7",
        "bg_tab_inactive":  "#1e3a5f",
        "accent":           "#4fc3f7",
        "accent_hover":     "#0288d1",
        "text_primary":     "#e0f7fa",
        "text_secondary":   "#80cbc4",
        "text_muted":       "#4db6ac",
        "border":           "#263859",
        "border_focus":     "#4fc3f7",
        "progress_bg":      "#263859",
        "progress_fill":    "#4fc3f7",
        "selection_bg":     "#4fc3f7",
        "selection_fg":     "#0d1b2a",
        "info_bg":          "#1e3a5f",
        "info_border":      "#263859",
        "tab_text_active":  "#0d1b2a",
        "tab_text_inactive":"#80cbc4",
    },
    "Solarized Dark": {
        "bg_main":          "#002b36",
        "bg_card":          "#073642",
        "bg_input":         "#004052",
        "bg_tab_active":    "#2aa198",
        "bg_tab_inactive":  "#004052",
        "accent":           "#2aa198",
        "accent_hover":     "#1a7a70",
        "text_primary":     "#eee8d5",
        "text_secondary":   "#93a1a1",
        "text_muted":       "#657b83",
        "border":           "#004052",
        "border_focus":     "#2aa198",
        "progress_bg":      "#004052",
        "progress_fill":    "#2aa198",
        "selection_bg":     "#2aa198",
        "selection_fg":     "#002b36",
        "info_bg":          "#004052",
        "info_border":      "#073642",
        "tab_text_active":  "#002b36",
        "tab_text_inactive":"#93a1a1",
    },
    "High Contrast": {
        "bg_main":          "#000000",
        "bg_card":          "#111111",
        "bg_input":         "#1a1a1a",
        "bg_tab_active":    "#ffff00",
        "bg_tab_inactive":  "#1a1a1a",
        "accent":           "#ffff00",
        "accent_hover":     "#cccc00",
        "text_primary":     "#ffffff",
        "text_secondary":   "#cccccc",
        "text_muted":       "#888888",
        "border":           "#444444",
        "border_focus":     "#ffff00",
        "progress_bg":      "#333333",
        "progress_fill":    "#ffff00",
        "selection_bg":     "#ffff00",
        "selection_fg":     "#000000",
        "info_bg":          "#1a1a1a",
        "info_border":      "#444444",
        "tab_text_active":  "#000000",
        "tab_text_inactive":"#cccccc",
    },
}


# ---------------------------------------------------------------------------
# Stylesheet builder
# ---------------------------------------------------------------------------

def build_stylesheet(theme_name: str = "Light (Default)", font_size: int = 10,
                     font_family: str = "Segoe UI", border_radius: int = 6,
                     compact: bool = False) -> str:
    """Generate a complete QSS stylesheet from a theme + layout settings."""
    t = THEMES.get(theme_name, THEMES["Light (Default)"])
    pad     = f"{4 if compact else 7}px {8 if compact else 14}px"
    tab_pad = f"{'8' if compact else '12'}px 12px"
    r = border_radius

    return f"""
        QMainWindow, QDialog, QWidget {{
            background-color: {t['bg_main']};
            color: {t['text_primary']};
            font-family: "{font_family}";
            font-size: {font_size}pt;
        }}
        QLabel {{
            color: {t['text_primary']};
        }}
        .title {{
            font-size: {font_size + 6}pt;
            font-weight: bold;
            color: {t['text_primary']};
        }}
        .subtitle, .section-subtitle, .stat-label {{
            font-size: {font_size - 1}pt;
            color: {t['text_secondary']};
        }}
        .section-title, .card-title, .stat-value {{
            font-size: {font_size + 1}pt;
            font-weight: 600;
            color: {t['text_primary']};
        }}

        /* ── Menu bar ──────────────────────────────────────────────────────── */
        QMenuBar {{
            background-color: {t['bg_card']};
            color: {t['text_primary']};
            border-bottom: 1px solid {t['border']};
        }}
        QMenuBar::item:selected {{
            background-color: {t['accent']};
            color: {t['tab_text_active']};
            border-radius: 4px;
        }}
        QMenu {{
            background-color: {t['bg_card']};
            color: {t['text_primary']};
            border: 1px solid {t['border']};
        }}
        QMenu::item:selected {{
            background-color: {t['selection_bg']};
            color: {t['selection_fg']};
        }}

        /* ── Buttons ───────────────────────────────────────────────────────── */
        QPushButton {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
            padding: {pad};
            font-size: {font_size - 1}pt;
            color: {t['text_primary']};
        }}
        QPushButton:hover {{
            background-color: {t['bg_input']};
            border-color: {t['border_focus']};
        }}
        QPushButton#primaryButton {{
            background-color: {t['accent']};
            color: {t['tab_text_active']};
            border: none;
        }}
        QPushButton#primaryButton:hover {{
            background-color: {t['accent_hover']};
        }}
        QPushButton#tabButton {{
            background-color: {t['bg_tab_inactive']};
            color: {t['tab_text_inactive']};
            border: none;
            border-bottom: 2px solid transparent;
            border-radius: 0px;
            padding: {tab_pad};
            font-size: {font_size}pt;
            font-weight: 500;
        }}
        QPushButton#tabButton:checked {{
            background-color: {t['bg_tab_active']};
            color: {t['tab_text_active']};
            border-bottom: 2px solid {t['accent']};
        }}
        QPushButton#tabButton:hover:!checked {{
            background-color: {t['bg_input']};
            color: {t['text_primary']};
        }}
        QPushButton#subtabPill {{
            background-color: {t['bg_main']};
            border: 1px solid {t['border']};
            border-radius: 12px;
            padding: 4px 12px;
            font-size: {font_size - 2}pt;
            font-weight: 500;
            color: {t['text_secondary']};
        }}
        QPushButton#subtabPill:checked {{
            background-color: {t['accent']};
            color: {t['tab_text_active']};
            border: 1px solid {t['accent']};
        }}
        QPushButton#subtabPill:hover:!checked {{
            background-color: {t['bg_input']};
            color: {t['text_primary']};
        }}

        /* ── Inputs ────────────────────────────────────────────────────────── */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {t['bg_input']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
            padding: {pad};
            color: {t['text_primary']};
            selection-background-color: {t['selection_bg']};
            selection-color: {t['selection_fg']};
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {t['border_focus']};
        }}
        QDoubleSpinBox, QSpinBox {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
            padding: 4px 28px 4px 8px;
            font-size: {font_size - 1}pt;
            color: {t['text_primary']};
            min-height: 12px;
        }}
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
            subcontrol-origin: border;
            width: 18px;
            border-left: 1px solid {t['border']};
            background: {t['bg_card']};
        }}
        /* Disabled controls previously rendered identically to enabled ones,
           so a greyed-out field looked editable and the toggle that governed
           it seemed to do nothing. Mute the text and background instead. */
        QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled,
        QComboBox:disabled, QPushButton:disabled, QCheckBox:disabled,
        QLabel:disabled, QSlider:disabled, QListWidget:disabled {{
            color: {t['text_muted']};
        }}
        QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled,
        QComboBox:disabled {{
            background-color: {t['bg_main']};
            border-color: {t['border']};
        }}
        QAbstractSpinBox::up-button:disabled,
        QAbstractSpinBox::down-button:disabled {{
            background: {t['bg_main']};
        }}
        QAbstractSpinBox::up-button {{
            subcontrol-position: top right;
            border-top-right-radius: {r}px;
        }}
        QAbstractSpinBox::down-button {{
            subcontrol-position: bottom right;
            border-bottom-right-radius: {r}px;
        }}
        QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {{
            width: 0px;
            height: 0px;
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 5px solid {t['text_secondary']};
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {t['text_secondary']};
        }}

        /* ── Combo box ─────────────────────────────────────────────────────── */
        QComboBox {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
            padding: 5px 10px;
            font-size: {font_size - 1}pt;
            color: {t['text_primary']};
            min-height: 16px;
        }}
        QComboBox:hover, QComboBox:focus {{
            border-color: {t['border_focus']};
        }}
        QComboBox::drop-down {{ border: none; width: 20px; }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {t['text_secondary']};
            margin-right: 8px;
        }}
        QComboBox QAbstractItemView {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            color: {t['text_primary']};
            selection-background-color: {t['selection_bg']};
            selection-color: {t['selection_fg']};
            padding: 4px;
        }}
        QComboBox QAbstractItemView::item {{ padding: 6px 10px; }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: {t['selection_bg']};
            color: {t['selection_fg']};
        }}

        /* ── Menus / lists ─────────────────────────────────────────────────── */
        QMenu, QListView {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            color: {t['text_primary']};
        }}
        QMenu::item, QListView::item {{ padding: 6px 12px; }}
        QMenu::item:selected, QListView::item:selected {{
            background-color: {t['selection_bg']};
            color: {t['selection_fg']};
        }}

        /* ── Sliders ───────────────────────────────────────────────────────── */
        QSlider::groove:horizontal {{
            border: none;
            height: 4px;
            background: {t['progress_bg']};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {t['accent']};
            border: none;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider::sub-page:horizontal {{
            background: {t['accent']};
            border-radius: 2px;
        }}

        /* ── Frames ────────────────────────────────────────────────────────── */
        QFrame#card {{
            background-color: {t['bg_card']};
            border-radius: {r + 4}px;
            border: 1px solid {t['border']};
        }}
        QFrame#infoBox {{
            background-color: {t['info_bg']};
            border: 1px solid {t['info_border']};
            border-radius: {r}px;
            padding: 12px;
        }}

        /* ── Group boxes ───────────────────────────────────────────────────── */
        QGroupBox {{
            background-color: {t['bg_card']};
            border: 1px solid {t['border']};
            border-radius: {r}px;
            margin-top: 12px;
            padding-top: 8px;
            font-weight: bold;
            color: {t['text_primary']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            top: -2px;
            color: {t['accent']};
        }}

        /* ── Checkboxes ────────────────────────────────────────────────────── */
        QCheckBox {{
            spacing: 8px;
            font-size: {font_size}pt;
            color: {t['text_primary']};
        }}
        QCheckBox::indicator {{
            width: 40px;
            height: 20px;
            border-radius: 10px;
            background-color: {t['border']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {t['accent']};
        }}

        /* ── Tab widget ────────────────────────────────────────────────────── */
        QTabWidget::pane {{
            border: none;
            top: -1px;
            background-color: {t['bg_card']};
        }}
        QTabBar::tab {{
            background: {t['bg_input']};
            color: {t['text_secondary']};
            padding: 8px 14px;
            border-top-left-radius: {r}px;
            border-top-right-radius: {r}px;
        }}
        QTabBar::tab:selected {{
            background: {t['bg_tab_active']};
            color: {t['tab_text_active']};
            font-weight: bold;
        }}

        /* ── Tables ────────────────────────────────────────────────────────── */
        QTableWidget {{
            background-color: {t['bg_card']};
            border: none;
            gridline-color: transparent;
            font-size: {font_size}pt;
            color: {t['text_primary']};
        }}
        QTableWidget::item {{
            padding: 8px;
            border-bottom: 1px solid {t['border']};
        }}
        QHeaderView::section {{
            background-color: {t['bg_card']};
            padding: 12px 8px;
            border: none;
            border-bottom: 1px solid {t['border']};
            font-weight: 600;
            font-size: {font_size - 1}pt;
            color: {t['text_secondary']};
        }}

        /* ── Progress bars ─────────────────────────────────────────────────── */
        QProgressBar {{
            border: none;
            border-radius: 4px;
            background-color: {t['progress_bg']};
            height: 8px;
        }}
        QProgressBar::chunk {{
            background-color: {t['progress_fill']};
            border-radius: 4px;
        }}

        /* ── Scroll bars ───────────────────────────────────────────────────── */
        QScrollBar:vertical {{
            background: {t['bg_main']};
            width: 8px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {t['border']};
            border-radius: 4px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {t['accent']}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{
            background: {t['bg_main']};
            height: 8px;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal {{
            background: {t['border']};
            border-radius: 4px;
            min-width: 20px;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {t['accent']}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """


def get_stylesheet() -> str:
    """
    Backwards-compatible entry point.
    Reads saved QSettings so the persisted theme is applied on startup.
    Falls back to the original Light default if no settings exist yet.
    """
    s = QSettings("MyCompany", "MixedSignalGUI")
    return build_stylesheet(
        theme_name    = s.value("theme",        "Light (Default)"),
        font_size     = int(s.value("fontSize",     10)),
        font_family   = s.value("fontFamily",   "Segoe UI"),
        border_radius = int(s.value("borderRadius", 6)),
        compact       = s.value("compact",      False, type=bool),
    )


def apply_theme_palette(theme_name: str = None):
    """
    Set the Qt application palette to match the theme colors.
    This is necessary for matplotlib plots to pick up theme changes.
    """
    from PySide6.QtWidgets import QApplication
    
    s = QSettings("MyCompany", "MixedSignalGUI")
    theme_name = theme_name or s.value("theme", "Light (Default)")
    t = THEMES.get(theme_name, THEMES["Light (Default)"])
    
    palette = QPalette()
    
    # Window colors (used by matplotlib for background)
    palette.setColor(QPalette.Window, QColor(t["bg_main"]))
    palette.setColor(QPalette.WindowText, QColor(t["text_primary"]))
    
    # Base colors (used by matplotlib for axes background)
    palette.setColor(QPalette.Base, QColor(t["bg_card"]))
    palette.setColor(QPalette.AlternateBase, QColor(t["bg_input"]))
    
    # Text colors
    palette.setColor(QPalette.Text, QColor(t["text_primary"]))
    palette.setColor(QPalette.BrightText, QColor(t["text_primary"]))
    
    # Button colors
    palette.setColor(QPalette.Button, QColor(t["bg_input"]))
    palette.setColor(QPalette.ButtonText, QColor(t["text_primary"]))
    
    # Highlight colors
    palette.setColor(QPalette.Highlight, QColor(t["selection_bg"]))
    palette.setColor(QPalette.HighlightedText, QColor(t["selection_fg"]))
    
    # Apply to application
    app = QApplication.instance()
    if app:
        app.setPalette(palette)


# ---------------------------------------------------------------------------
# Settings Dialog  (lives here so stylesheet.py is the single source of truth)
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    """
    Tabbed settings dialog.
      • Appearance  – theme, font family/size, border radius, compact mode
      • Paths       – model / data / output folders
      • Compute     – CPU/GPU, GPU memory, mixed precision, workers, MATLAB
      • Plots       – DPI, colour map, auto-refresh, antialiasing
    """

    settingsChanged = Signal()   # emitted on Apply / OK so main window can re-theme

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self.setModal(True)
        self._settings = QSettings("MyCompany", "MixedSignalGUI")
        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------ UI --

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)

        tabs = QTabWidget()
        tabs.addTab(self._appearance_tab(), "🎨  Appearance")
        tabs.addTab(self._paths_tab(),      "📁  Paths")
        tabs.addTab(self._compute_tab(),    "⚙️  Compute")
        tabs.addTab(self._plot_tab(),       "📊  Plots")
        root.addWidget(tabs)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        root.addWidget(line)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset defaults")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()

        btn_apply = QPushButton("Apply")
        btn_apply.clicked.connect(self._apply)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self._ok)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        root.addLayout(btn_row)

    def _appearance_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        grp = QGroupBox("Theme")
        form = QFormLayout(grp)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEMES.keys()))
        form.addRow("Colour theme:", self.theme_combo)

        self.swatch = QLabel("  Preview  ")
        self.swatch.setAutoFillBackground(True)
        self.swatch.setAlignment(Qt.AlignCenter)
        self.swatch.setFixedHeight(26)
        form.addRow("", self.swatch)
        self.theme_combo.currentTextChanged.connect(self._update_swatch)

        layout.addWidget(grp)

        grp2 = QGroupBox("Typography")
        form2 = QFormLayout(grp2)
        self.font_combo = QFontComboBox()
        form2.addRow("Font family:", self.font_combo)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(7, 18)
        self.font_size_spin.setSuffix(" pt")
        form2.addRow("Font size:", self.font_size_spin)
        layout.addWidget(grp2)

        grp3 = QGroupBox("Layout")
        form3 = QFormLayout(grp3)
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(0, 16)
        self.radius_slider.setTickInterval(4)
        self.radius_slider.setTickPosition(QSlider.TicksBelow)
        self.radius_label = QLabel("6 px")
        self.radius_slider.valueChanged.connect(
            lambda v: self.radius_label.setText(f"{v} px")
        )
        rl = QHBoxLayout()
        rl.addWidget(self.radius_slider)
        rl.addWidget(self.radius_label)
        form3.addRow("Border radius:", rl)
        self.compact_check = QCheckBox("Compact mode (smaller padding)")
        form3.addRow("", self.compact_check)
        layout.addWidget(grp3)

        layout.addStretch()
        return w

    def _paths_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        grp = QGroupBox("File Paths")
        form = QFormLayout(grp)

        for attr, label, placeholder in [
            ("model_path_edit", "Model folder:",  "Select model folder…"),
            ("data_path_edit",  "Data folder:",   "Select data folder…"),
            ("output_path_edit","Output folder:", "Select output folder…"),
        ]:
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            setattr(self, attr, edit)
            btn = QPushButton("Browse…")
            btn.clicked.connect(lambda _, e=edit: self._browse(e))
            hl = QHBoxLayout()
            hl.addWidget(edit)
            hl.addWidget(btn)
            form.addRow(label, hl)

        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _compute_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        grp = QGroupBox("Compute")
        form = QFormLayout(grp)

        self.compute_mode_combo = QComboBox()
        self.compute_mode_combo.addItems(["CPU", "GPU"])
        form.addRow("Compute mode:", self.compute_mode_combo)

        self.experiment_seed_spin = QSpinBox()
        self.experiment_seed_spin.setRange(0, 2**31 - 1)
        self.experiment_seed_spin.setToolTip("Default seed for waveform generation, channel augmentation, and training. Applies to new runs.")
        form.addRow("Experiment seed:", self.experiment_seed_spin)

        self.gpu_mem_spin = QDoubleSpinBox()
        self.gpu_mem_spin.setRange(0.1, 1.0)
        self.gpu_mem_spin.setSingleStep(0.1)
        self.gpu_mem_spin.setDecimals(1)
        self.gpu_mem_spin.setSuffix("  (fraction)")
        form.addRow("GPU memory fraction:", self.gpu_mem_spin)

        self.mixed_prec_check = QCheckBox("Enable mixed precision (FP16)")
        form.addRow("", self.mixed_prec_check)

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(0, 32)
        form.addRow("DataLoader workers:", self.num_workers_spin)

        layout.addWidget(grp)

        grp2 = QGroupBox("MATLAB Engine")
        form2 = QFormLayout(grp2)
        self.matlab_lazy_check = QCheckBox("Lazy-start MATLAB engine (start on first use)")
        form2.addRow("", self.matlab_lazy_check)
        layout.addWidget(grp2)

        layout.addStretch()
        return w

    def _plot_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        grp = QGroupBox("Plot Defaults")
        form = QFormLayout(grp)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 300)
        self.dpi_spin.setSuffix(" dpi")
        form.addRow("Figure DPI:", self.dpi_spin)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "cividis",
            "jet", "hot", "cool", "gray", "Blues", "Reds",
        ])
        form.addRow("Default colour map:", self.cmap_combo)

        self.auto_refresh_check = QCheckBox("Auto-refresh plots when data changes")
        form.addRow("", self.auto_refresh_check)

        self.antialiasing_check = QCheckBox("Enable antialiasing")
        form.addRow("", self.antialiasing_check)

        layout.addWidget(grp)
        layout.addStretch()
        return w

    # ── Helpers ────────────────────────────────────────────────────────────

    def _browse(self, widget: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select folder")
        if path:
            widget.setText(path)

    def _update_swatch(self, theme_name: str):
        t = THEMES.get(theme_name, THEMES["Dark"])
        self.swatch.setStyleSheet(
            f"background-color: {t['bg_main']}; color: {t['text_primary']}; "
            f"border: 2px solid {t['accent']}; border-radius: 4px;"
        )
        self.swatch.setText(f"  {theme_name}  ")

    # ── Persistence ────────────────────────────────────────────────────────

    def _load_values(self):
        s = self._settings
        self.theme_combo.setCurrentText(s.value("theme",        "Light (Default)"))
        self.font_combo.setCurrentFont(QFont(s.value("fontFamily",   "Segoe UI")))
        self.font_size_spin.setValue(   int(s.value("fontSize",     10)))
        self.radius_slider.setValue(    int(s.value("borderRadius",  6)))
        self.compact_check.setChecked(  s.value("compact",      False, type=bool))
        self.model_path_edit.setText(   s.value("modelPath",  ""))
        self.data_path_edit.setText(    s.value("dataPath",   ""))
        self.output_path_edit.setText(  s.value("outputPath", ""))
        self.compute_mode_combo.setCurrentText(s.value("mode", "CPU"))
        self.experiment_seed_spin.setValue(int(s.value("experimentSeed", 42)))
        self.gpu_mem_spin.setValue(     float(s.value("gpuMemFraction", 0.9)))
        self.mixed_prec_check.setChecked(s.value("mixedPrecision", False, type=bool))
        self.num_workers_spin.setValue( int(s.value("numWorkers", 4)))
        self.matlab_lazy_check.setChecked(s.value("matlabLazy", True, type=bool))
        self.dpi_spin.setValue(         int(s.value("plotDPI",  100)))
        self.cmap_combo.setCurrentText( s.value("colourMap", "viridis"))
        self.auto_refresh_check.setChecked(s.value("autoRefresh",   True,  type=bool))
        self.antialiasing_check.setChecked(s.value("antialiasing",  True,  type=bool))
        self._update_swatch(self.theme_combo.currentText())

    def _save_values(self):
        s = self._settings
        s.setValue("theme",          self.theme_combo.currentText())
        s.setValue("fontFamily",     self.font_combo.currentFont().family())
        s.setValue("fontSize",       self.font_size_spin.value())
        s.setValue("borderRadius",   self.radius_slider.value())
        s.setValue("compact",        self.compact_check.isChecked())
        s.setValue("modelPath",      self.model_path_edit.text())
        s.setValue("dataPath",       self.data_path_edit.text())
        s.setValue("outputPath",     self.output_path_edit.text())
        s.setValue("mode",           self.compute_mode_combo.currentText())
        s.setValue("experimentSeed", self.experiment_seed_spin.value())
        s.setValue("gpuMemFraction", self.gpu_mem_spin.value())
        s.setValue("mixedPrecision", self.mixed_prec_check.isChecked())
        s.setValue("numWorkers",     self.num_workers_spin.value())
        s.setValue("matlabLazy",     self.matlab_lazy_check.isChecked())
        s.setValue("plotDPI",        self.dpi_spin.value())
        s.setValue("colourMap",      self.cmap_combo.currentText())
        s.setValue("autoRefresh",    self.auto_refresh_check.isChecked())
        s.setValue("antialiasing",   self.antialiasing_check.isChecked())

    def _apply(self):
        self._save_values()
        self.settingsChanged.emit()

    def _ok(self):
        self._apply()
        self.accept()

    def _reset_defaults(self):
        self.theme_combo.setCurrentText("Light (Default)")
        self.font_combo.setCurrentFont(QFont("Segoe UI"))
        self.font_size_spin.setValue(10)
        self.radius_slider.setValue(6)
        self.compact_check.setChecked(False)
        self.compute_mode_combo.setCurrentText("CPU")
        self.experiment_seed_spin.setValue(42)
        self.gpu_mem_spin.setValue(0.9)
        self.mixed_prec_check.setChecked(False)
        self.num_workers_spin.setValue(4)
        self.matlab_lazy_check.setChecked(True)
        self.dpi_spin.setValue(100)
        self.cmap_combo.setCurrentText("viridis")
        self.auto_refresh_check.setChecked(True)
        self.antialiasing_check.setChecked(True)
