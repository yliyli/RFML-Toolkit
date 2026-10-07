"""Explicit, non-overwriting result exports grouped by model/data contents."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

import numpy as np


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def array_fingerprint(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def data_identity_from_hashes(hashes, labels, classes):
    labels = np.asarray(labels).tolist()
    return {"sha256": _digest([hashes, labels, list(classes)]),
            "sample_count": len(hashes), "classes": list(classes)}


def data_identity(arrays, labels=(), classes=()):
    return data_identity_from_hashes([array_fingerprint(a) for a in arrays], labels, classes)


def model_identity(model, metadata, checkpoint):
    import torch
    weights = [(name, str(tensor.dtype), list(tensor.shape),
                array_fingerprint(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()))
               for name, tensor in sorted(model.state_dict().items())]
    return {"sha256": _digest([weights, metadata]),
            "checkpoint": str(Path(checkpoint).absolute()),
            "metadata": metadata}


def output_folder():
    from PySide6.QtCore import QSettings
    configured = QSettings("MyCompany", "MixedSignalGUI").value("outputPath", "")
    return Path(str(configured).strip()) if str(configured or '').strip() else Path.home() / 'Documents' / 'rfml_output'


def export_run(root, *, data, model=None, kind, description, figures=None,
               tables=None, texts=None, details=None):
    """Write PNGs/CSVs into a new run directory; index only completed exports.

    A data fingerprint includes native arrays, ordered labels, and class names.
    A model fingerprint includes loaded weights and companion metadata. Paths
    document provenance but do not determine pair identity. No samples are changed.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    pair_id = _digest([data['sha256'], (model or {}).get('sha256')])
    group = root / f"{'evaluation' if model else 'visualization'}_{pair_id[:20]}"
    group.mkdir(exist_ok=True)
    run_name = f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%S_%fZ}_{uuid.uuid4().hex[:8]}"
    staging = group / f".pending_{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / 'plots').mkdir()
        (staging / 'csv').mkdir()
        for name, figure in (figures or {}).items():
            figure.savefig(staging / 'plots' / f'{name}.png', dpi=150, bbox_inches='tight')
        for name, (header, rows) in (tables or {}).items():
            with (staging / 'csv' / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        for name, text in (texts or {}).items():
            (staging / f'{name}.txt').write_text(str(text), encoding='utf-8')
        info = {"kind": kind, "description": description, "data": data,
                "model": model, "details": details or {}, "timestamp_utc": run_name}
        (staging / 'run.json').write_text(json.dumps(info, indent=2, default=str), encoding='utf-8')
        (staging / 'run.txt').write_text(
            f"{kind}\n{description}\nModel: {(model or {}).get('checkpoint', 'none')}\n"
            f"Samples: {data['sample_count']}\n"
            "PNG images: plots/; numeric tables: csv/; full provenance: run.json\n"
            "Plot units and calibration are unchanged; these exports are not new measurements.\n",
            encoding='utf-8')
        staging.rename(group / run_name)
    except Exception:
        # Only the private staging folder created by this export is removed.
        shutil.rmtree(staging)
        raise
    pair_file = group / 'pair.json'
    if not pair_file.exists():
        pair_file.write_text(json.dumps({"data": data, "model": model,
                                        "description": description}, indent=2, default=str), encoding='utf-8')
    lines = ["RFML result exports\n", "Groups use content fingerprints, not filenames.\n",
             "Each run_* folder is a separate export; timestamps are UTC.\n\n"]
    for p in sorted(root.glob('*/pair.json')):
        identity = json.loads(p.read_text())
        lines.append(f"{p.parent.name}\n  Model: {(identity['model'] or {}).get('checkpoint', 'none (data visualization)')}\n"
                     f"  Data: {identity['description']}\n  Data SHA256: {identity['data']['sha256']}\n"
                     f"  Runs: {len(list(p.parent.glob('run_*')))}\n\n")
    index_tmp = root / f'.index_{uuid.uuid4().hex}.tmp'
    index_tmp.write_text(''.join(lines), encoding='utf-8')
    index_tmp.replace(root / 'index.txt')
    return group / run_name


def export_with_dialog(parent, **payload):
    from PySide6.QtWidgets import QMessageBox
    try:
        path = export_run(output_folder(), **payload)
    except Exception as exc:
        QMessageBox.warning(parent, "Export Failed", str(exc))
        return None
    QMessageBox.information(parent, "Results Exported", f"Saved results to:\n{path}")
    return path
