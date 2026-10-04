"""Persistent experiment defaults shared by the GUI and workers."""

from PySide6.QtCore import QSettings


def experiment_seed():
    return int(QSettings("MyCompany", "MixedSignalGUI").value("experimentSeed", 42))
