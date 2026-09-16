"""Deterministic typed-array artifacts for the direct baseline."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np


def deterministic_npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name or name.endswith(".npy"):
                raise ValueError(f"invalid NPZ member name: {name!r}")
            array = np.asarray(arrays[name])
            if array.dtype.hasobject:
                raise ValueError(f"object arrays are forbidden: {name}")
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, member.getvalue(), compresslevel=9)
    return output.getvalue()


def write_npz_exclusive(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(deterministic_npz_bytes(arrays))
    observed = load_npz(path)
    if observed.keys() != arrays.keys():
        raise RuntimeError("baseline artifact inventory changed on round-trip")
    for name, expected in arrays.items():
        if not np.array_equal(observed[name], expected, equal_nan=True):
            raise RuntimeError(f"baseline artifact changed on round-trip: {name}")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


__all__ = ["deterministic_npz_bytes", "load_npz", "write_npz_exclusive"]
