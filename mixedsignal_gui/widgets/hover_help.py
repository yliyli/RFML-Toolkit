"""Non-modal help that remains readable while the pointer is over its panel."""

from PySide6.QtCore import QEvent, QPoint, QTimer, Qt
from PySide6.QtWidgets import QFrame, QLabel, QToolButton, QVBoxLayout


CHECKPOINT_HELP = (
    "<b>Load a trained model</b><br>Choose a toolbox .pth checkpoint. Keep its "
    "companion JSON beside it: this records class labels, input length, channels, "
    "and preprocessing. For a custom model, also keep the saved Python plugin "
    "beside the checkpoint. Only load trusted checkpoints and Python files."
)


class HoverHelpButton(QToolButton):
    def __init__(self, guidance, parent=None):
        super().__init__(parent)
        self.setText("?")
        self.setAccessibleName("Help")
        self.setAccessibleDescription(guidance)
        self.setFixedSize(24, 24)
        self.setStyleSheet(
            "QToolButton { border: 1px solid palette(mid); border-radius: 12px; "
            "padding: 0; color: palette(window-text); background: palette(window); }"
            "QToolButton:hover { border-color: palette(highlight); }"
        )
        self.popup = QFrame(self, Qt.ToolTip)
        self.popup.setAttribute(Qt.WA_ShowWithoutActivating)
        self.popup.setFrameShape(QFrame.StyledPanel)
        self.popup.setAutoFillBackground(True)
        layout = QVBoxLayout(self.popup)
        self.help_label = QLabel(guidance)
        self.help_label.setTextFormat(Qt.RichText)
        self.help_label.setWordWrap(True)
        self.help_label.setFixedWidth(360)
        layout.addWidget(self.help_label)
        self.popup.installEventFilter(self)
        self.help_label.installEventFilter(self)
        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.setInterval(150)  # Allow crossing the small gap to the panel.
        self.hide_timer.timeout.connect(self._hide_if_outside)
        self.clicked.connect(self.show_help)  # Also usable with keyboard/click.

    def show_help(self):
        self.hide_timer.stop()
        self.popup.adjustSize()
        position = self.mapToGlobal(QPoint(0, self.height() + 2))
        screen = self.screen().availableGeometry()
        x = max(screen.left(), min(position.x(), screen.right() - self.popup.width() + 1))
        y = position.y()
        if y + self.popup.height() > screen.bottom():
            y = self.mapToGlobal(QPoint(0, 0)).y() - self.popup.height() - 2
        self.popup.move(x, max(screen.top(), y))
        self.popup.show()

    def enterEvent(self, event):
        self.show_help()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hide_timer.start()
        super().leaveEvent(event)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            self.hide_timer.stop()
        elif event.type() == QEvent.Leave:
            self.hide_timer.start()
        return super().eventFilter(obj, event)

    def _hide_if_outside(self):
        if not self.underMouse() and not self.popup.underMouse():
            self.popup.hide()

    def hideEvent(self, event):
        self.hide_timer.stop()
        self.popup.hide()
        super().hideEvent(event)
