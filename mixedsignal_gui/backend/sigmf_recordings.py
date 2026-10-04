"""Read observed I/Q regions from conforming, single-channel SigMF recordings.

This reader never estimates a channel, changes the sample rate, or pads samples.
"""

import json
from pathlib import Path
import re

import numpy as np


def read_sigmf_regions(path):
    """Return (samples, metadata) pairs for annotated regions or whole captures.

    Supported types: cf32/cf64 and ci8/ci16/ci32, with declared byte order.
    Multi-channel and non-conforming datasets with headers are rejected.
    """
    path = Path(path).expanduser().resolve()
    if path.suffix not in (".sigmf-meta", ".sigmf-data"):
        raise ValueError("Select a .sigmf-meta or .sigmf-data file.")
    meta_path = path.with_suffix(".sigmf-meta")
    data_path = path.with_suffix(".sigmf-data")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    global_meta = meta.get("global", {})
    fs = global_meta.get("core:sample_rate")
    if fs is None or not np.isfinite(float(fs)) or float(fs) <= 0:
        raise ValueError("SigMF requires a positive core:sample_rate.")
    if global_meta.get("core:num_channels", 1) != 1:
        raise ValueError("Only single-channel SigMF recordings are supported.")
    if global_meta.get("core:offset", 0) or global_meta.get("core:trailing_bytes", 0):
        raise ValueError("SigMF datasets with offsets or trailing bytes are not supported.")
    if global_meta.get("core:dataset"):
        raise ValueError("Select a conforming .sigmf-data pair; external core:dataset paths are not supported.")
    datatype = global_meta.get("core:datatype", "")
    match = re.fullmatch(r"c(f32|f64|i8|i16|i32)(?:_(le|be))?", datatype)
    if not match or (match.group(1) != "i8" and not match.group(2)):
        raise ValueError(f"Unsupported complex SigMF datatype: {datatype!r}.")
    base, endian = match.groups()
    numpy_type = {"f32": "c8", "f64": "c16", "i8": "i1", "i16": "i2", "i32": "i4"}[base]
    dtype = np.dtype((">" if endian == "be" else "<") + numpy_type)
    sample_bytes = dtype.itemsize * (2 if base.startswith("i") else 1)
    size = data_path.stat().st_size
    if not size or size % sample_bytes:
        raise ValueError("SigMF data is empty or contains an incomplete I/Q sample.")
    total = size // sample_bytes
    raw = np.memmap(data_path, mode="r", dtype=dtype)

    def sample_index(value, field):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer.")
        return value

    captures = meta.get("captures") or [{"core:sample_start": 0}]
    captures = sorted(captures, key=lambda c: sample_index(c.get("core:sample_start"), "Capture start"))
    starts = [c["core:sample_start"] for c in captures]
    if len(set(starts)) != len(starts) or starts[-1] >= total:
        raise ValueError("SigMF capture boundaries are duplicated or outside the recording.")
    if any(c.get("core:header_bytes", 0) for c in captures):
        raise ValueError("SigMF data with capture headers is not supported.")
    # An omitted initial capture implies no frequency information before it.
    if starts[0] > 0:
        captures.insert(0, {"core:sample_start": 0})
        starts.insert(0, 0)
    ends = starts[1:] + [total]
    annotations = meta.get("annotations") or []
    regions = []
    for annotation in annotations:
        start = sample_index(annotation.get("core:sample_start"), "Annotation start")
        capture_index = int(np.searchsorted(starts, start, side="right")) - 1
        if not 0 <= start < total:
            raise ValueError("SigMF annotation starts outside the recording.")
        count = annotation.get("core:sample_count", ends[capture_index] - start)
        count = sample_index(count, "Annotation sample count")
        if count <= 0 or start + count > total:
            raise ValueError("SigMF annotation has an empty or out-of-bounds sample region.")
        regions.append((start, start + count, annotation))
    if not annotations:
        regions = [(start, end, {}) for start, end in zip(starts, ends)]

    result = []
    for region_start, region_end, annotation in regions:
        for capture, start, end in zip(captures, starts, ends):
            lo, hi = max(region_start, start), min(region_end, end)
            if lo >= hi:
                continue
            if base.startswith("i"):
                # Preserve integer amplitudes, including ci32 precision.
                iq = np.asarray(raw[2 * lo:2 * hi], dtype=np.float64)
                samples = iq[0::2] + 1j * iq[1::2]
            else:
                samples = np.array(raw[lo:hi], dtype=np.complex64 if base == "f32" else np.complex128)
            metadata = {
                "source": "sigmf_recording", "fs": float(fs),
                "samples": hi - lo, "output_type": "baseband",
                "source_path": str(data_path), "sigmf_meta_path": str(meta_path),
                "sample_start": lo, "sample_count": hi - lo,
                "sigmf": {"global": global_meta, "capture": capture,
                          "annotation": annotation},
            }
            label = annotation.get("core:label") or capture.get("core:label")
            if label:
                metadata["modulation"] = str(label)
                metadata["class_label"] = str(label)
            if capture.get("core:frequency") is not None:
                metadata["fc"] = float(capture["core:frequency"])
            result.append((samples, metadata))
    return result
