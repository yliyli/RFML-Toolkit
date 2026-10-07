"""
backend/dataset_manager.py

Disk-based dataset registry. The datasets/ folder is the single source of
truth. Every dataset is a pair of files:
    <name>.npy   – the raw signal (NumPy array)
    <name>.json  – metadata / config sidecar

All tabs interact with DatasetManager to list, load, save, and import
datasets. Nothing is stored in memory beyond the currently-active selection,
which is owned by the UI (the combo box), not by this class.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Minimal required metadata keys for a valid sidecar
# ---------------------------------------------------------------------------
_REQUIRED_KEYS = {"name", "source", "fs", "samples"}


def dataset_folder_path(root, name):
    """Validate a user-supplied immediate subfolder name without creating it."""
    name = name.strip()
    if not name or name in ('.', '..') or any(c in name for c in '/\\\0:'):
        raise ValueError("Enter a folder name, not a path (no /, \\, or colon).")
    root = Path(root).resolve()
    target = root / name
    if target.resolve().parent != root:
        raise ValueError("The dataset folder must stay inside the selected data folder.")
    if target.exists() and not target.is_dir():
        raise ValueError("That name already belongs to a file.")
    return target


def _validate(meta: dict) -> bool:
    return _REQUIRED_KEYS.issubset(meta.keys())


class DatasetManager:
    """
    Scan-on-demand, disk-backed registry of signal datasets.

    Usage
    -----
    dm = DatasetManager(datasets_dir)
    entries = dm.scan()           # list[dict]  – all metadata sidecars
    signal  = dm.load_signal(entry)  # np.ndarray
    dm.save(name, signal, metadata)  # writes .npy + .json
    dm.import_external(npy_path, metadata)  # copies into datasets_dir
    """

    def __init__(self, datasets_dir: Optional[str] = None):
        if datasets_dir is None:
            # Fall back to <project_root>/datasets/ if no dir given
            here = Path(__file__).resolve().parent
            datasets_dir = str(here.parent / "datasets")
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def scan(self) -> list[dict]:
        """
        Return a list of metadata dicts for every valid .json sidecar found
        in datasets_dir. Entries without a matching .npy are skipped.
        Results are sorted newest-first by timestamp (falling back to name).
        """
        entries = []
        for json_path in sorted(self.datasets_dir.glob("*.json")):
            npy_path = json_path.with_suffix(".npy")
            if not npy_path.exists():
                continue
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                if not _validate(meta):
                    print(f"[DatasetManager] Skipping {json_path.name}: missing required keys")
                    continue
                # Always store absolute paths so callers don't have to know the dir
                meta["_json_path"] = str(json_path)
                meta["_npy_path"]  = str(npy_path)
                entries.append(meta)
            except Exception as e:
                print(f"[DatasetManager] Could not read {json_path.name}: {e}")

        # Sort newest-first; fall back to alphabetical by name
        entries.sort(
            key=lambda m: m.get("timestamp", m.get("name", "")),
            reverse=True,
        )
        return entries

    def load_signal(self, entry: dict) -> np.ndarray:
        """Load and return the numpy array for a metadata entry from scan()."""
        npy_path = entry.get("_npy_path") or str(
            self.datasets_dir / (entry["name"] + ".npy")
        )
        return np.load(npy_path, allow_pickle=False)

    def save(self, name: str, signal: np.ndarray, metadata: dict) -> dict:
        """
        Write signal + sidecar JSON into datasets_dir.

        Parameters
        ----------
        name     : dataset name (used as filename stem)
        signal   : NumPy array to save as <name>.npy
        metadata : dict – must include at minimum: fs, source, samples

        Returns the completed metadata dict (with _npy_path / _json_path).
        """
        meta = dict(metadata)
        meta["name"] = name
        meta.setdefault("samples", int(signal.shape[-1]) if hasattr(signal, 'shape') else int(len(signal)))
        meta.setdefault("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

        if not _validate(meta):
            missing = _REQUIRED_KEYS - meta.keys()
            raise ValueError(f"Metadata missing required keys: {missing}")

        npy_path  = self.datasets_dir / f"{name}.npy"
        json_path = self.datasets_dir / f"{name}.json"

        np.save(str(npy_path), signal)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        meta["_npy_path"]  = str(npy_path)
        meta["_json_path"] = str(json_path)

        print(f"[DatasetManager] Saved: {npy_path.name} + {json_path.name}")
        return meta

    def import_external(
        self,
        npy_path: str,
        metadata: dict,
        *,
        copy: bool = True,
    ) -> dict:
        """
        Register an external .npy file as a dataset.

        Parameters
        ----------
        npy_path : path to the source .npy file
        metadata : dict with at least: fs, source='imported'
                   (name will be derived from filename if not given)
        copy     : if True, copy the file into datasets_dir (default).
                   Pass False to reference in-place (not recommended).

        Returns the completed metadata dict.
        """
        src = Path(npy_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {npy_path}")

        name = metadata.get("name") or src.stem
        # Avoid collisions
        name = self._unique_name(name)

        if copy:
            dst = self.datasets_dir / f"{name}.npy"
            shutil.copy2(str(src), str(dst))
        else:
            dst = src

        signal = np.load(str(dst), allow_pickle=False)

        meta = dict(metadata)
        meta.setdefault("source", "imported")
        meta.setdefault("fs", 8e6)     # caller should always supply this
        meta["samples"] = int(signal.shape[-1]) if hasattr(signal, 'shape') else int(len(signal))

        return self.save(name, signal, meta)

    def import_sigmf(self, path: str) -> list[dict]:
        """Store observed SigMF regions unchanged in the shared dataset registry."""
        from .sigmf_recordings import read_sigmf_regions
        regions = read_sigmf_regions(path)
        stem = Path(path).stem
        entries = []
        for index, (samples, metadata) in enumerate(regions):
            name = self._unique_name(f"{stem}_region{index}")
            entries.append(self.save(name, samples, metadata))
        return entries

    def delete(self, name: str) -> bool:
        """Delete both files for a dataset. Returns True if deleted."""
        npy  = self.datasets_dir / f"{name}.npy"
        json_ = self.datasets_dir / f"{name}.json"
        deleted = False
        for p in (npy, json_):
            if p.exists():
                p.unlink()
                deleted = True
        return deleted

    # ------------------------------------------------------------------
    # Convenience helpers used by the UI
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        """Sorted list of dataset names available on disk."""
        return [e["name"] for e in self.scan()]

    def get_by_name(self, name: str) -> Optional[dict]:
        """Return the metadata entry for a given name, or None."""
        for entry in self.scan():
            if entry["name"] == name:
                return entry
        return None

    def _unique_name(self, base: str) -> str:
        """Append _1, _2 … until the name is not taken."""
        name = base
        counter = 1
        while (self.datasets_dir / f"{name}.npy").exists():
            name = f"{base}_{counter}"
            counter += 1
        return name
