from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

DEFAULT_N_JOBS = 4
MAX_N_JOBS = 4


def candidate_set_hash(candidate_ids: list[str]) -> str:
    payload = ",".join(sorted(candidate_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cap_n_jobs(requested: int = -1, *, default: int = DEFAULT_N_JOBS, maximum: int = MAX_N_JOBS) -> int:
    cpu = os.cpu_count() or default
    upper = min(maximum, cpu)
    if requested <= 0:
        return min(default, upper)
    return max(1, min(requested, upper))


def memory_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def log_progress(stage: str, done: int, total: int, start_time: float) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta_s = remaining / rate if rate > 0 else 0.0
    pct = (100.0 * done / total) if total else 100.0
    print(
        f"[{stage}] {done}/{total} ({pct:.1f}%) "
        f"elapsed={elapsed:.1f}s eta={eta_s:.1f}s mem={memory_mb():.0f}MB",
        flush=True,
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))


def atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def atomic_write_numpy(path: Path, array: Any) -> None:
    import numpy as np

    path = Path(path)
    if path.suffix != ".npy":
        path = path.with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = path.parent / f"{path.stem}.tmp"
    np.save(str(tmp_base), array)
    tmp_file = Path(f"{tmp_base}.npy")
    tmp_file.replace(path)


def atomic_copy_file(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def atomic_write_via_temp(path: Path, writer: Callable[[Path], None]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    writer(tmp)
    tmp.replace(path)
