#!/usr/bin/env python3
"""
ARIES Stage 8.3F — process-isolated full-frame PyRaDoN FRT scan.

Each individual find_single() call runs in a fresh child process.
The parent owns checkpointing and disk-backed working images.

Scientific configuration:
- Stage 8.2 PSF-filtered image.
- Stage 8.1 empirical variance map.
- Empirical PSF.
- Pinned PyRaDoN commit.
- Full-length, full-frame FRT.
- Both orientations.
- Native descending threshold schedule.
- Three iterations per orientation.
- Native Streak footprint.

After each candidate, the exact Streak footprint is zeroed in both
the working image and variance map, following the recommendation
in Nir et al. 2018, Section 4.4, footnote 6.

This is a candidate-generation run, not final false-alarm calibration.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import resource
import subprocess
import sys
import time
import traceback

from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Limit numerical-library threading before importing NumPy/SciPy
# ---------------------------------------------------------------------------

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


import numpy as np
import psutil


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

EXPECTED_COMMIT = (
    "6a1a43096b30ba99cdfea5593dc5cad0f35d36a3"
)

DEFAULT_PROJECT = (
    Path("/home/deepmalya/pip_malya/ILMT_Streak_Detection")
)

DEFAULT_BASE = (
    DEFAULT_PROJECT
    / "outputs/single_frame_preprocessing/20221026-0057"
)

DEFAULT_IMAGE = (
    DEFAULT_BASE
    / "frt_full_length_empirical_psf_v2/initial_psf_filtered_image.npy"
)

DEFAULT_VARIANCE = (
    DEFAULT_BASE
    / "frt_empirical_variance_proxy.npy"
)

DEFAULT_PSF = (
    DEFAULT_BASE
    / "frt_empirical_psf_kernel.npy"
)

DEFAULT_REPO = (
    DEFAULT_PROJECT / "pyradon_reference"
)

DEFAULT_RUN_DIR = (
    DEFAULT_BASE
    / "frt_full_length_empirical_psf_v2"
    / "native_threshold_scan_isolated_v1"
)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def atomic_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def sha256_file(
    path: Path,
    block_size: int = 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        for block in iter(
            lambda: f.read(block_size),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def memory_snapshot() -> dict[str, Any]:

    vm = psutil.virtual_memory()

    proc = psutil.Process()

    out = {

        "host_total_gib": round(
            vm.total / 1024**3,
            3,
        ),

        "host_available_gib": round(
            vm.available / 1024**3,
            3,
        ),

        "host_used_percent": round(
            vm.percent,
            1,
        ),

        "parent_rss_gib": round(
            proc.memory_info().rss / 1024**3,
            3,
        ),

    }

    # cgroup v2

    root = Path("/sys/fs/cgroup")

    max_file = root / "memory.max"

    cur_file = root / "memory.current"

    if max_file.exists() and cur_file.exists():

        raw = max_file.read_text().strip()

        current = int(
            cur_file.read_text().strip()
        )

        if raw != "max":

            limit = int(raw)

            out["cgroup_limit_gib"] = round(
                limit / 1024**3,
                3,
            )

            out["cgroup_current_gib"] = round(
                current / 1024**3,
                3,
            )

            out["cgroup_headroom_gib"] = round(
                max(0, limit - current) / 1024**3,
                3,
            )

        else:

            out["cgroup_limit_gib"] = "unlimited"

    # cgroup v1 fallback

    if "cgroup_limit_gib" not in out:

        v1_max = Path(
            "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        )

        v1_cur = Path(
            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        )

        if v1_max.exists() and v1_cur.exists():

            limit = int(
                v1_max.read_text().strip()
            )

            current = int(
                v1_cur.read_text().strip()
            )

            if limit < 2**60:

                out["cgroup_limit_gib"] = round(
                    limit / 1024**3,
                    3,
                )

                out["cgroup_current_gib"] = round(
                    current / 1024**3,
                    3,
                )

                out["cgroup_headroom_gib"] = round(
                    max(0, limit - current) / 1024**3,
                    3,
                )

    return out


def require_memory_headroom(
    args: argparse.Namespace,
    phase: str,
) -> dict[str, Any]:

    mem = memory_snapshot()

    host_avail = float(
        mem["host_available_gib"]
    )

    if host_avail < float(
        args.min_host_available_gib
    ):

        raise RuntimeError(
            f"Insufficient host RAM before {phase}: "
            f"{host_avail:.2f} GiB available; "
            f"require {args.min_host_available_gib:.1f} GiB. "
            "The checkpoint is safe; close other applications "
            "or increase memory/swap, then resume."
        )

    cg_headroom = mem.get(
        "cgroup_headroom_gib"
    )

    if (
        isinstance(cg_headroom, (int, float))
        and cg_headroom < float(
            args.min_cgroup_headroom_gib
        )
    ):

        raise RuntimeError(
            f"Insufficient cgroup headroom before {phase}: "
            f"{cg_headroom:.2f} GiB; "
            f"require {args.min_cgroup_headroom_gib:.1f} GiB. "
            "Host RAM alone is misleading here."
        )

    return mem


def current_git_commit(
    repo: Path,
) -> str:

    p = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if p.returncode:

        raise RuntimeError(
            "Cannot read PyRaDoN Git commit: "
            + p.stderr.strip()
        )

    return p.stdout.strip()


def python_source_tree_sha256(
    repo: Path,
) -> str:

    h = hashlib.sha256()

    for source_path in sorted(
        repo.rglob("*.py")
    ):

        if (
            ".git" in source_path.parts
            or "__pycache__" in source_path.parts
        ):
            continue

        h.update(
            source_path.relative_to(repo)
            .as_posix()
            .encode("utf-8")
        )

        h.update(b"\0")

        with source_path.open("rb") as source_handle:

            for block in iter(
                lambda: source_handle.read(
                    1024 * 1024
                ),
                b"",
            ):

                h.update(block)

    return h.hexdigest()


def copy_npy(
    src_path: Path,
    dst_path: Path,
    dtype=np.float32,
    rows: int = 128,
) -> None:

    src = np.load(
        src_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if src.ndim != 2:

        raise ValueError(
            f"Expected 2-D array in {src_path}; got {src.shape}"
        )

    dst_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = np.lib.format.open_memmap(
        dst_path,
        mode="w+",
        dtype=dtype,
        shape=src.shape,
    )

    for r0 in range(
        0,
        src.shape[0],
        rows,
    ):

        r1 = min(
            r0 + rows,
            src.shape[0],
        )

        out[r0:r1] = np.asarray(
            src[r0:r1],
            dtype=dtype,
        )

    out.flush()

    del out, src


def chunked_mean_and_subtract(
    arr: np.memmap,
    variance: np.ndarray | np.memmap | None = None,
    rows: int = 128,
) -> float:

    """
    Subtract mean over active, positive-variance pixels only.

    Previously masked image and variance pixels remain zero.
    """

    total = 0.0

    count = 0

    for r0 in range(
        0,
        arr.shape[0],
        rows,
    ):

        block = np.asarray(
            arr[r0:r0 + rows]
        )

        active = np.isfinite(block)

        if variance is not None:

            vb = np.asarray(
                variance[r0:r0 + rows]
            )

            active &= (
                np.isfinite(vb)
                & (vb > 0)
            )

        if active.any():

            total += float(
                np.sum(
                    block[active],
                    dtype=np.float64,
                )
            )

            count += int(
                active.sum()
            )

    if count == 0:

        raise RuntimeError(
            "No active pixels remain while subtracting the mean."
        )

    mean = total / count

    for r0 in range(
        0,
        arr.shape[0],
        rows,
    ):

        r1 = min(
            r0 + rows,
            arr.shape[0],
        )

        block = arr[r0:r1]

        active = np.isfinite(block)

        if variance is not None:

            vb = np.asarray(
                variance[r0:r1]
            )

            active &= (
                np.isfinite(vb)
                & (vb > 0)
            )

        block[active] -= np.float32(mean)

    arr.flush()

    return float(mean)


def atomic_checkpoint_state(
    state_path: Path,
    state: dict[str, Any],
) -> None:

    atomic_json(
        state_path,
        state,
    )


def advance_after_call(
    state: dict[str, Any],
    was_candidate: bool,
    max_iter: int,
) -> None:

    state["frt_calls"] = int(
        state.get("frt_calls", 0)
    ) + 1

    state["pending_call"] = None

    if was_candidate:

        state["iteration_index"] = (
            int(state["iteration_index"]) + 1
        )

        if state["iteration_index"] >= max_iter:

            state["orientation_index"] = (
                int(state["orientation_index"]) + 1
            )

            state["iteration_index"] = 0

    else:

        # Like native find_multi(): stop this orientation
        # at the first miss.

        state["orientation_index"] = (
            int(state["orientation_index"]) + 1
        )

        state["iteration_index"] = 0


def worker_script_path() -> Path:

    return Path(__file__).resolve()

# ---------------------------------------------------------------------------
# Stage 8.2 AST compatibility patch
#
# Reproduces the audited full-length patch:
#   1. Call get_norm_factor_psf()
#   2. Select geometric factor [:, 0, :]
#   3. Use nanargmax
#   4. Guard all-invalid Radon maps
# ---------------------------------------------------------------------------

def _is_self_attr(
    node: ast.AST,
    attr: str,
) -> bool:

    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr == attr
    )


def _is_self_call(
    node: ast.AST,
    attr: str,
) -> bool:

    return (
        isinstance(node, ast.Call)
        and _is_self_attr(node.func, attr)
    )


def _targets_name(
    stmt: ast.stmt,
    name: str,
) -> bool:

    if isinstance(stmt, ast.Assign):

        targets = stmt.targets

    elif isinstance(
        stmt,
        (ast.AnnAssign, ast.AugAssign),
    ):

        targets = [stmt.target]

    else:

        return False

    return any(
        isinstance(t, ast.Name) and t.id == name
        for t in targets
    )


def _is_radon_argmax(
    node: ast.AST,
) -> bool:

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "np"
        and node.func.attr in {"argmax", "nanargmax"}
        and len(node.args) >= 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "radon_image"
    )


def _has_all_invalid_guard(
    stmt: ast.stmt,
) -> bool:

    if not isinstance(stmt, ast.If):
        return False

    try:

        condition = ast.unparse(
            stmt.test
        )

    except Exception:

        return False

    if condition != "not np.isfinite(radon_image).any()":
        return False

    return any(
        isinstance(n, ast.Return)
        and n.value is None
        for n in ast.walk(
            ast.Module(
                body=stmt.body,
                type_ignores=[],
            )
        )
    )


def _patch_block(
    statements: list[ast.stmt],
    changes: list[str],
    found: dict[str, bool],
) -> list[ast.stmt]:

    output = []

    for stmt in statements:

        for field in (
            "body",
            "orelse",
            "finalbody",
        ):

            child = getattr(
                stmt,
                field,
                None,
            )

            if isinstance(child, list):

                setattr(
                    stmt,
                    field,
                    _patch_block(
                        child,
                        changes,
                        found,
                    ),
                )

        for handler in getattr(
            stmt,
            "handlers",
            [],
        ):

            handler.body = _patch_block(
                handler.body,
                changes,
                found,
            )

        for case in getattr(
            stmt,
            "cases",
            [],
        ):

            case.body = _patch_block(
                case.body,
                changes,
                found,
            )

        # PSF normalisation method must be called.

        if (
            _targets_name(stmt, "psf_factor")
            and hasattr(stmt, "value")
        ):

            value = stmt.value

            if _is_self_attr(
                value,
                "get_norm_factor_psf",
            ):

                stmt.value = ast.Call(
                    func=value,
                    args=[],
                    keywords=[],
                )

                changes.append(
                    "called get_norm_factor_psf()"
                )

                found["psf"] = True

            elif _is_self_call(
                value,
                "get_norm_factor_psf",
            ):

                found["psf"] = True

        # Geometric factor must be flattened to the
        # correct 2-D long-mode plane.

        if (
            _targets_name(stmt, "geom_factor")
            and hasattr(stmt, "value")
        ):

            value = stmt.value

            if _is_self_call(
                value,
                "get_geometric_factor",
            ):

                stmt.value = ast.Subscript(

                    value=value,

                    slice=ast.Tuple(
                        elts=[
                            ast.Slice(),
                            ast.Constant(value=0),
                            ast.Slice(),
                        ],
                        ctx=ast.Load(),
                    ),

                    ctx=ast.Load(),

                )

                changes.append(
                    "selected geometric factor [:, 0, :]"
                )

                found["geometry"] = True

            elif (
                isinstance(value, ast.Subscript)
                and _is_self_call(
                    value.value,
                    "get_geometric_factor",
                )
            ):

                found["geometry"] = True

        # NaN-safe maximum selection.

        if (
            _targets_name(stmt, "idx")
            and hasattr(stmt, "value")
        ):

            value = stmt.value

            if _is_radon_argmax(value):

                if value.func.attr == "argmax":

                    value.func.attr = "nanargmax"

                    changes.append(
                        "replaced np.argmax with np.nanargmax"
                    )

                found["argmax"] = True

                if not (
                    output
                    and _has_all_invalid_guard(
                        output[-1]
                    )
                ):

                    output.append(
                        ast.parse(
                            "if not np.isfinite(radon_image).any():\n"
                            "    return None"
                        ).body[0]
                    )

                    changes.append(
                        "added all-invalid Radon-map guard"
                    )

        output.append(stmt)

    return output


def patch_finder_long_mode(
    Finder: type,
) -> None:

    module = inspect.getmodule(Finder)

    if (
        module is None
        or not getattr(module, "__file__", None)
    ):

        raise RuntimeError(
            "Cannot locate imported pinned PyRaDoN Finder module."
        )

    source_path = Path(
        module.__file__
    ).resolve()

    source = source_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(source_path),
    )

    classes = [
        n for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name == Finder.__name__
    ]

    if len(classes) != 1:

        raise RuntimeError(
            "Cannot uniquely locate Finder class in pinned finder.py"
        )

    methods = [
        n for n in classes[0].body
        if isinstance(n, ast.FunctionDef)
        and n.name == "find_single"
    ]

    if len(methods) != 1:

        raise RuntimeError(
            "Cannot uniquely locate Finder.find_single in pinned source"
        )

    method = copy.deepcopy(
        methods[0]
    )

    method.decorator_list = []

    changes = []

    found = {
        "psf": False,
        "geometry": False,
        "argmax": False,
    }

    method.body = _patch_block(
        method.body,
        changes,
        found,
    )

    dump = ast.dump(
        method,
        include_attributes=False,
    )

    has_long = (
        "radon_variance_maps" in dump
        and "get_geometric_factor" in dump
        and "radon_image" in dump
    )

    if not all(found.values()) or not has_long:

        raise RuntimeError(
            "Pinned source does not match audited Stage 8.2 patch: "
            f"found={found}, long_mode={has_long}"
        )

    method.name = "_frt_patched_find_single"

    patch_module = ast.Module(
        body=[method],
        type_ignores=[],
    )

    ast.fix_missing_locations(
        patch_module
    )

    exec(
        compile(
            patch_module,
            str(source_path),
            "exec",
        ),
        module.__dict__,
    )

    patched = module.__dict__.pop(
        "_frt_patched_find_single"
    )

    patched.__name__ = "find_single"

    patched.__qualname__ = (
        f"{Finder.__name__}.find_single"
    )

    Finder.find_single = patched

    sig = inspect.signature(
        Finder.find_single
    )

    if not all(
        k in sig.parameters
        for k in (
            "im",
            "transpose",
            "threshold",
        )
    ):

        raise RuntimeError(
            f"Unexpected patched find_single signature: {sig}"
        )


# ---------------------------------------------------------------------------
# Child worker: one FRT calculation, then exit
# ---------------------------------------------------------------------------

def child_worker(
    args: argparse.Namespace,
) -> int:

    start = time.perf_counter()

    process = psutil.Process(os.getpid())

    def log_worker_memory(stage: str) -> None:
        try:
            rss_gib = process.memory_info().rss / 1024**3
            peak_gib = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**2
            )
            print(
                f"[WORKER_MEMORY] {stage}: "
                f"rss_gib={rss_gib:.3f}, "
                f"peak_rss_gib={peak_gib:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WORKER_MEMORY] {stage}: logging failed: {exc}",
                flush=True,
            )

    log_worker_memory("worker start")

    result_path = Path(
        args.result_json
    )

    try:

        repo = Path(
            args.repo
        ).resolve()

        if current_git_commit(repo) != args.commit:

            raise RuntimeError(
                f"PyRaDoN commit mismatch: expected {args.commit}, "
                f"found {current_git_commit(repo)}"
            )

        actual_source_hash = python_source_tree_sha256(
            repo
        )

        if (
            args.source_sha256
            and actual_source_hash != args.source_sha256
        ):

            raise RuntimeError(
                "PyRaDoN Python source hash mismatch: "
                f"expected {args.source_sha256}, "
                f"found {actual_source_hash}"
            )

        sys.path.insert(
            0,
            str(repo),
        )

        from src.finder import Finder

        patch_finder_long_mode(
            Finder
        )

        image = np.load(
            args.work,
            mmap_mode="r",
            allow_pickle=False,
        )

        variance = np.load(
            args.variance,
            mmap_mode="r",
            allow_pickle=False,
        )

        psf = np.load(
            args.psf,
            allow_pickle=False,
        )

        if (
            image.shape != (4096, 4096)
            or image.dtype != np.float32
        ):

            raise RuntimeError(
                f"Unexpected working image: "
                f"shape={image.shape}, dtype={image.dtype}"
            )

        if (
            variance.shape != image.shape
            or variance.dtype != np.float32
        ):

            raise RuntimeError(
                f"Unexpected variance: "
                f"shape={variance.shape}, dtype={variance.dtype}"
            )

        if (
            psf.ndim != 2
            or not np.isfinite(psf).all()
            or not np.isclose(
                float(psf.sum()),
                1.0,
                atol=1e-6,
            )
        ):

            raise RuntimeError(
                "Invalid saved empirical PSF kernel."
            )

        f = Finder(

            threshold=float(
                args.min_threshold
            ),

            num_iterations=int(
                args.num_iterations
            ),

            min_length=int(
                args.min_length
            ),

            use_short=False,

            use_sections=False,

            use_exclude=False,

            use_subtract_mean=True,

            use_conv=False,

            verbosity=0,

        )

        f.data.variance = variance

        f.data.psf = np.array(
            psf,
            dtype=np.float64,
            copy=True,
        )

        f.data.image = image

        f.data._current_section_corner = (
            0,
            0,
        )

        f.pars.use_short = False

        f.pars.use_sections = False

        f.pars.use_show = False

        f.pars.use_subtract_mean = True

        # Image was already PSF-convolved in Stage 8.2.
        # Do not apply a second convolution.

        f.pars.use_conv = False

        # The variance map can contain zeroed candidate footprints after
        # streak removal. Its median is therefore not required to equal
        # the original frame-level clipping scalar.
        finder_var_scalar = float(f.data.var_scalar)

        if not np.isfinite(finder_var_scalar):
            raise RuntimeError(
                "Finder variance summary is non-finite."
            )

        var_check = f.data.get_var_image(
            (0, 0),
            image.shape,
        )

        if var_check.shape != image.shape:

            raise RuntimeError(
                "Finder returned an unexpected variance-map shape."
            )

        if (
            not np.isfinite(var_check).all()
            or np.any(var_check < 0)
        ):

            raise RuntimeError(
                "Variance map contains invalid/negative values."
            )

        # ---------------------------------------------------------------
        # Execute exactly one full-frame FRT call
        # ---------------------------------------------------------------

        log_worker_memory("before find_single")

        streak = f.find_single(

            image,

            transpose=bool(
                args.transpose
            ),

            threshold=float(
                args.threshold
            ),

        )

        log_worker_memory("after find_single")

        if streak is None:

            payload = {

                "status": "no_candidate",

                "transpose": bool(args.transpose),

                "threshold": float(args.threshold),

                "elapsed_seconds": float(
                    time.perf_counter() - start
                ),

                "worker_maxrss_gib": round(
                    resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss / 1024**2,
                    3,
                ),

            }

            atomic_json(
                result_path,
                payload,
            )

            return 0

        def val(name: str) -> float | None:

            try:

                x = float(
                    getattr(streak, name)
                )

            except (
                AttributeError,
                TypeError,
                ValueError,
                OverflowError,
            ):

                return None

            return (
                x if np.isfinite(x)
                else None
            )

        x1 = val("x1f")

        y1 = val("y1f")

        x2 = val("x2f")

        y2 = val("y2f")

        if None in (
            x1,
            y1,
            x2,
            y2,
        ):

            raise RuntimeError(
                "PyRaDoN candidate lacks finite full-frame endpoints."
            )

        record = {

            "x1_px": x1,

            "y1_px": y1,

            "x2_px": x2,

            "y2_px": y2,

            "length_px": float(
                np.hypot(
                    x2 - x1,
                    y2 - y1,
                )
            ),

            "image_angle_deg": float(
                np.degrees(
                    np.arctan2(
                        y2 - y1,
                        x2 - x1,
                    )
                ) % 180.0
            ),

            "pyradon_angle_deg": val("th"),

            "pyradon_snr": val("snr"),

            "pyradon_snr_fwhm": val("snr_fwhm"),

            "intensity_per_length": val("I"),

            "streak_length_px": val("L"),

        }

        # ---------------------------------------------------------------
        # Prepare transactional image and variance updates
        # ---------------------------------------------------------------

        out_work = Path(
            args.output_work
        )

        out_var = Path(
            args.output_variance
        )

        out_work.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        out_var.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_work = out_work.with_suffix(
            out_work.suffix + ".tmp.npy"
        )

        temp_var = out_var.with_suffix(
            out_var.suffix + ".tmp.npy"
        )

        if temp_work.exists():
            temp_work.unlink()

        if temp_var.exists():
            temp_var.unlink()

        # Build exact PyRaDoN Streak model footprint.

        streak_module = inspect.getmodule(
            type(streak)
        )

        model_fn = (
            getattr(streak_module, "model", None)
            if streak_module
            else None
        )

        if not callable(model_fn):

            raise RuntimeError(
                "Cannot locate pinned Streak.model() function."
            )

        log_worker_memory("before chunked model mask")

        mask = np.empty(image.shape, dtype=bool)

        chunk_rows = 32
        halo_rows = 1

        for r0 in range(0, image.shape[0], chunk_rows):
            r1 = min(r0 + chunk_rows, image.shape[0])

            # Include neighbouring rows to preserve the original
            # model's convolution/downsampling behaviour at boundaries.
            ext0 = max(0, r0 - halo_rows)
            ext1 = min(image.shape[0], r1 + halo_rows)

            local_mask = model_fn(
                (int(ext1 - ext0), int(image.shape[1])),
                streak.x1,
                streak.x2,
                streak.y1 - ext0,
                streak.y2 - ext0,
                streak.psf_sigma,
            ) > 0

            expected_shape = (ext1 - ext0, image.shape[1])

            if local_mask.shape != expected_shape:
                raise RuntimeError(
                    f"Chunked streak model has shape {local_mask.shape}; "
                    f"expected {expected_shape}."
                )

            crop0 = r0 - ext0
            crop1 = r1 - ext0

            mask[r0:r1, :] = local_mask[crop0:crop1, :]
            del local_mask

        log_worker_memory("after chunked model mask")

        if mask.shape != image.shape:

            raise RuntimeError(
                f"Streak mask shape {mask.shape} != image shape {image.shape}."
            )

        masked_count = int(
            np.count_nonzero(mask)
        )

        if masked_count == 0:

            raise RuntimeError(
                "Detected streak produced an empty subtraction mask."
            )

        # ---------------------------------------------------------------
        # Copy the current image to a new versioned file
        # ---------------------------------------------------------------

        writable = np.lib.format.open_memmap(

            temp_work,

            mode="w+",

            dtype=np.float32,

            shape=image.shape,

        )

        rows = 128

        for r0 in range(
            0,
            image.shape[0],
            rows,
        ):

            writable[r0:r0 + rows] = np.asarray(
                image[r0:r0 + rows],
                dtype=np.float32,
            )

        # Reuse the already generated exact Streak footprint.
        # Do not call streak.subtract_streak(): it would regenerate
        # the model over the entire full-resolution image.
        writable[mask] = 0.0

        log_worker_memory("after image mask update")

        if np.any(writable[mask] != 0.0):
            raise RuntimeError(
                "Direct mask application did not zero the expected footprint."
            )

        # ---------------------------------------------------------------
        # Update variance over exactly the same footprint
        # ---------------------------------------------------------------

        var_out = np.lib.format.open_memmap(

            temp_var,

            mode="w+",

            dtype=np.float32,

            shape=variance.shape,

        )

        for r0 in range(
            0,
            variance.shape[0],
            rows,
        ):

            r1 = min(
                r0 + rows,
                variance.shape[0],
            )

            vb = np.array(
                variance[r0:r1],
                dtype=np.float32,
                copy=True,
            )

            vb[mask[r0:r1]] = 0.0

            var_out[r0:r1] = vb

        var_out.flush()

        log_worker_memory("after variance mask update")

        # Exclude zero-variance pixels from mean calculation.

        if bool(args.subtract_mean):

            chunked_mean_and_subtract(

                writable,

                variance=var_out,

                rows=rows,

            )

        writable.flush()

        del writable
        del var_out
        del mask

        os.replace(
            temp_work,
            out_work,
        )

        os.replace(
            temp_var,
            out_var,
        )

        payload = {

            "status": "candidate",

            "transpose": bool(args.transpose),

            "threshold": float(args.threshold),

            "record": record,

            "output_work": str(out_work),

            "output_variance": str(out_var),

            "variance_masked_pixel_count": int(
                masked_count
            ),

            "variance_mask_update": (
                "zero image and variance at exact Streak model footprint; "
                "Nir et al. 2018 Section 4.4, footnote 6"
            ),

            "elapsed_seconds": float(
                time.perf_counter() - start
            ),

            "worker_maxrss_gib": round(
                resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss / 1024**2,
                3,
            ),

        }

        atomic_json(
            result_path,
            payload,
        )

        return 0

    except BaseException as exc:

        payload = {

            "status": "worker_error",

            "error": repr(exc),

            "traceback": traceback.format_exc(),

            "elapsed_seconds": float(
                time.perf_counter() - start
            ),

            "worker_maxrss_gib": round(
                resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss / 1024**2,
                3,
            ),

        }

        try:

            atomic_json(
                result_path,
                payload,
            )

        except Exception:

            pass

        return 2
# ---------------------------------------------------------------------------
# Parent process: checkpointed scan orchestration
# ---------------------------------------------------------------------------

def scan_parent(
    args: argparse.Namespace,
) -> int:

    image_path = Path(
        args.image
    ).resolve()

    variance_path = Path(
        args.variance
    ).resolve()

    psf_path = Path(
        args.psf
    ).resolve()

    repo = Path(
        args.repo
    ).resolve()

    run_dir = Path(
        args.run_dir
    ).resolve()

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Validate input files and pinned repository
    # ---------------------------------------------------------------

    for p in (
        image_path,
        variance_path,
        psf_path,
    ):

        if not p.is_file():

            raise FileNotFoundError(
                f"Required saved FRT product not found: {p}"
            )

    if current_git_commit(repo) != args.commit:

        raise RuntimeError(
            f"Expected PyRaDoN commit {args.commit}, "
            f"got {current_git_commit(repo)}"
        )

    source_tree_hash = python_source_tree_sha256(
        repo
    )

    mem = memory_snapshot()

    print(
        "STAGE 8.3F — PROCESS-ISOLATED FULL-FRAME FRT"
    )

    print(
        "--------------------------------------------"
    )

    print("Image:", image_path)

    print("Variance:", variance_path)

    print("PSF:", psf_path)

    print(
        "PyRaDoN repo/commit:",
        repo,
        args.commit,
    )

    print(
        "Memory preflight:",
        mem,
    )

    print(
        "Mode: full-length, full-frame; "
        "each FRT call runs in a fresh process"
    )

    print(
        "Clip mode:",
        args.clip_mode,
    )

    print(
        "Variance mask after candidate: enabled"
    )

    # Never force a run below the memory safety threshold.

    require_memory_headroom(
        args,
        "initial FRT scan",
    )

    # ---------------------------------------------------------------
    # Load memory-mapped input arrays
    # ---------------------------------------------------------------

    initial = np.load(
        image_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    variance = np.load(
        variance_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    psf = np.load(
        psf_path,
        allow_pickle=False,
    )

    if (
        initial.shape != (4096, 4096)
        or initial.dtype != np.float32
    ):

        raise ValueError(
            f"Initial image must be (4096,4096) float32; "
            f"got {initial.shape} {initial.dtype}"
        )

    if (
        variance.shape != initial.shape
        or variance.dtype != np.float32
    ):

        raise ValueError(
            f"Variance map must match image as float32; "
            f"got {variance.shape} {variance.dtype}"
        )

    if not np.isfinite(initial).all():

        raise ValueError(
            "Initial PSF-filtered image contains non-finite values."
        )

    if (
        not np.isfinite(variance).all()
        or np.any(variance <= 0)
    ):

        raise ValueError(
            "Initial variance map must be finite and strictly positive."
        )

    if (
        psf.ndim != 2
        or not np.isfinite(psf).all()
        or np.any(psf < 0)
        or not np.isclose(
            float(psf.sum()),
            1.0,
            atol=1e-6,
        )
    ):

        raise ValueError(
            "Saved empirical PSF is invalid."
        )

    var_scalar = float(
        args.var_scalar
    )

    if (
        not np.isfinite(var_scalar)
        or var_scalar <= 0
    ):

        raise ValueError(
            "var_scalar must be positive and finite."
        )

    # ---------------------------------------------------------------
    # Calculate maximum pointwise S/N in row blocks
    # ---------------------------------------------------------------

    max_snr = -np.inf

    rows = 128

    for r0 in range(
        0,
        initial.shape[0],
        rows,
    ):

        r1 = min(
            r0 + rows,
            initial.shape[0],
        )

        snr = (
            np.asarray(initial[r0:r1])
            /
            np.sqrt(
                np.asarray(variance[r0:r1])
            )
        )

        finite = np.isfinite(
            snr
        )

        if finite.any():

            max_snr = max(
                max_snr,
                float(
                    np.max(snr[finite])
                ),
            )

        del snr
        del finite

    if (
        not np.isfinite(max_snr)
        or max_snr <= args.min_threshold
    ):

        raise RuntimeError(
            f"Invalid maximum pointwise S/N: {max_snr}"
        )

    dynamic_range = math.log2(
        max_snr / args.min_threshold
    )

    thresholds = np.flip(

        args.min_threshold
        *
        2 ** np.arange(
            dynamic_range + 1
        )

    ).astype(float).tolist()

    print(
        "Maximum pointwise S/N:",
        max_snr,
    )

    print(
        "Dynamic range:",
        dynamic_range,
    )

    print(
        "Threshold levels:",
        len(thresholds),
        thresholds,
    )

    print(
        "Worst-case FRT calls:",
        len(thresholds)
        * 2
        * args.num_iterations,
    )

    # ---------------------------------------------------------------
    # Run signature
    # ---------------------------------------------------------------

    signature = {

        "frame": args.frame,

        "image_path": str(image_path),

        "image_sha256_file": sha256_file(
            image_path
        ),

        "variance_path": str(variance_path),

        "variance_sha256_file": sha256_file(
            variance_path
        ),

        "psf_path": str(psf_path),

        "psf_sha256_file": sha256_file(
            psf_path
        ),

        "repo": str(repo),

        "commit": args.commit,

        "python_source_tree_sha256": source_tree_hash,

        "var_scalar": var_scalar,

        "minimum_radon_snr": float(
            args.min_threshold
        ),

        "thresholds": thresholds,

        "iterations_per_orientation": int(
            args.num_iterations
        ),

        "minimum_streak_length": int(
            args.min_length
        ),

        "use_short": False,

        "use_sections": False,

        "mean_subtraction": True,

        "psf_already_applied": True,

        "clip_mode": args.clip_mode,

        "variance_mask_update_after_streak_removal": True,

        "orchestration": (
            "one child process per single full-frame find_single call; "
            "versioned disk checkpoints"
        ),

    }

    # Release parent's input mappings before launching workers.

    del initial
    del variance
    del psf

    gc.collect()

    print(
        "Parent input mappings released; "
        "memory before FRT workers:",
        memory_snapshot(),
    )

    # ---------------------------------------------------------------
    # Checkpoint paths
    # ---------------------------------------------------------------

    state_path = run_dir / "scan_state.json"

    csv_path = run_dir / "frt_candidates.csv"

    summary_path = run_dir / "run_summary.json"

    # ---------------------------------------------------------------
    # Load or initialize scan state
    # ---------------------------------------------------------------

    if state_path.exists():

        state = json.loads(
            state_path.read_text(
                encoding="utf-8"
            )
        )

        if state.get("signature") != signature:

            raise RuntimeError(
                "Existing isolated-run checkpoint signature differs. "
                "Use a new --run-dir."
            )

        print(
            "RESUMING isolated run at threshold",
            int(state["next_threshold_index"]) + 1,
        )

    else:

        work_name = "work_initial.npy"

        variance_name = "variance_initial.npy"

        copy_npy(
            image_path,
            run_dir / work_name,
        )

        copy_npy(
            variance_path,
            run_dir / variance_name,
        )

        state = {

            "signature": signature,

            "thresholds": thresholds,

            "next_threshold_index": 0,

            "clipped_for_level": False,

            "orientation_index": 0,

            "iteration_index": 0,

            "work_file": work_name,

            "variance_file": variance_name,

            "candidates": [],

            "frt_calls": 0,

            "pending_call": None,

            "complete": False,

            "started_unix": time.time(),

        }

        atomic_checkpoint_state(
            state_path,
            state,
        )

        print(
            "Created isolated checkpoint:",
            state_path,
        )

    # ---------------------------------------------------------------
    # Candidate CSV writer
    # ---------------------------------------------------------------

    def save_csv() -> None:

        cols = [

            "candidate_id",

            "threshold_index",

            "threshold_snr",

            "orientation_transpose",

            "iteration_in_orientation",

            "x1_px",

            "y1_px",

            "x2_px",

            "y2_px",

            "length_px",

            "image_angle_deg",

            "pyradon_angle_deg",

            "pyradon_snr",

            "pyradon_snr_fwhm",

            "intensity_per_length",

            "streak_length_px",

            "variance_masked_pixel_count",

            "worker_maxrss_gib",

            "worker_elapsed_seconds",

        ]

        tmp = csv_path.with_suffix(
            ".csv.tmp"
        )

        with tmp.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=cols,
                extrasaction="ignore",
            )

            w.writeheader()

            w.writerows(
                state["candidates"]
            )

        os.replace(
            tmp,
            csv_path,
        )

    # ---------------------------------------------------------------
    # Commit worker result transactionally
    # ---------------------------------------------------------------

    def commit_child_result(
        pending: dict[str, Any],
        result: dict[str, Any],
    ) -> None:

        old_work_file = state.get(
            "work_file"
        )

        old_variance_file = state.get(
            "variance_file"
        )

        if result.get("status") == "worker_error":

            raise RuntimeError(
                "FRT child failed:\n"
                + str(
                    result.get(
                        "traceback",
                        result.get("error"),
                    )
                )
            )

        is_candidate = (
            result.get("status") == "candidate"
        )

        if result.get("status") not in {
            "candidate",
            "no_candidate",
        }:

            raise RuntimeError(
                f"Unexpected child result: {result}"
            )

        if is_candidate:

            out_work = Path(
                result["output_work"]
            )

            out_var = Path(
                result["output_variance"]
            )

            if (
                not out_work.is_file()
                or not out_var.is_file()
            ):

                raise RuntimeError(
                    "Candidate worker reported success "
                    "but output image/variance is missing."
                )

            record = dict(
                result["record"]
            )

            record.update({

                "candidate_id": (
                    len(state["candidates"]) + 1
                ),

                "threshold_index": int(
                    pending["threshold_index"]
                ),

                "threshold_snr": float(
                    pending["threshold"]
                ),

                "orientation_transpose": bool(
                    pending["transpose"]
                ),

                "iteration_in_orientation": (
                    int(pending["iteration_index"]) + 1
                ),

                "variance_masked_pixel_count": int(
                    result.get(
                        "variance_masked_pixel_count",
                        0,
                    )
                ),

                "worker_maxrss_gib": float(
                    result.get(
                        "worker_maxrss_gib",
                        np.nan,
                    )
                ),

                "worker_elapsed_seconds": float(
                    result.get(
                        "elapsed_seconds",
                        np.nan,
                    )
                ),

            })

            state["candidates"].append(
                record
            )

            state["work_file"] = (
                out_work.name
            )

            state["variance_file"] = (
                out_var.name
            )

        advance_after_call(
            state,
            is_candidate,
            int(args.num_iterations),
        )

        atomic_checkpoint_state(
            state_path,
            state,
        )

        save_csv()

        # Remove older version only after checkpoint commits.

        for old_name, new_name in (

            (
                old_work_file,
                state.get("work_file"),
            ),

            (
                old_variance_file,
                state.get("variance_file"),
            ),

        ):

            if old_name and old_name != new_name:

                try:

                    (
                        run_dir / old_name
                    ).unlink()

                except FileNotFoundError:

                    pass

        print(

            f"  call {int(state['frt_calls']):03d} "
            f"| transpose={bool(pending['transpose'])} "
            f"| iter={int(pending['iteration_index'])+1}/{args.num_iterations} "
            f"| candidate={'yes' if is_candidate else 'no'} "
            f"| child_peak_RSS={result.get('worker_maxrss_gib')} GiB "
            f"| child_s={result.get('elapsed_seconds')} "
            f"| candidates={len(state['candidates'])}")

    # ---------------------------------------------------------------
    # Reconcile interrupted worker
    # ---------------------------------------------------------------

    if state.get("pending_call"):

        pending = state["pending_call"]

        rpath = Path(
            pending["result_json"]
        )

        if rpath.exists():

            existing = json.loads(
                rpath.read_text(
                    encoding="utf-8"
                )
            )

            if existing.get("status") == "worker_error":

                print(
                    "Previous worker error:",
                    existing.get("error"),
                )

                print(
                    existing.get(
                        "traceback",
                        "",
                    )
                )

                # Clear failed pending result so the same
                # uncommitted FRT call can be retried.

                rpath.unlink(
                    missing_ok=True
                )

                state["pending_call"] = None

                atomic_checkpoint_state(
                    state_path,
                    state,
                )

                print(
                    "Failed worker result discarded. "
                    "Checkpoint image was not advanced."
                )

            else:

                commit_child_result(
                    pending,
                    existing,
                )

                state["pending_call"] = None

                atomic_checkpoint_state(
                    state_path,
                    state,
                )

                rpath.unlink(
                    missing_ok=True
                )

        else:

            # Worker was killed before returning a result.
            # No image update was committed.

            state["pending_call"] = None

            atomic_checkpoint_state(
                state_path,
                state,
            )

            print(
                "Previous worker did not produce a result. "
                "The same FRT call will be retried."
            )

    # ---------------------------------------------------------------
    # Main scan
    # ---------------------------------------------------------------

    start_time = time.perf_counter()

    calls_this_run = 0

    while int(
        state["next_threshold_index"]
    ) < len(thresholds):

        level = int(
            state["next_threshold_index"]
        )

        threshold = float(
            thresholds[level]
        )

        work_path = (
            run_dir / state["work_file"]
        )

        var_path = (
            run_dir / state["variance_file"]
        )

        # -----------------------------------------------------------
        # Apply threshold clipping once per level
        # -----------------------------------------------------------

        if not state["clipped_for_level"]:

            new_name = (
                f"work_after_clip_{level:02d}.npy"
            )

            new_path = (
                run_dir / new_name
            )

            src = np.load(
                work_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            var = np.load(
                var_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            dst = np.lib.format.open_memmap(

                new_path,

                mode="w+",

                dtype=np.float32,

                shape=src.shape,

            )

            sqrt_scalar = math.sqrt(
                var_scalar
            )

            for r0 in range(
                0,
                src.shape[0],
                rows,
            ):

                r1 = min(
                    r0 + rows,
                    src.shape[0],
                )

                ib = np.asarray(
                    src[r0:r1],
                    dtype=np.float32,
                )

                vb = np.asarray(
                    var[r0:r1],
                    dtype=np.float32,
                )

                sig = np.sqrt(
                    vb,
                    dtype=np.float32,
                )

                valid = (
                    np.isfinite(ib)
                    & np.isfinite(sig)
                    & (sig > 0)
                )

                snr = np.full(
                    ib.shape,
                    -np.inf,
                    dtype=np.float32,
                )

                np.divide(
                    ib,
                    sig,
                    out=snr,
                    where=valid,
                )

                clip = (
                    valid
                    & (snr > threshold / 2.0)
                )

                ob = np.array(
                    ib,
                    dtype=np.float32,
                    copy=True,
                )

                if args.clip_mode == "pyradon_scalar":

                    ob[clip] = np.float32(
                        (threshold / 2.0)
                        * sqrt_scalar
                    )

                else:

                    cap = (
                        np.float32(threshold / 2.0)
                        * sig
                    )

                    ob[clip] = cap[clip]

                dst[r0:r1] = ob

            dst.flush()

            del dst
            del src
            del var

            # Match native scan mean subtraction.
            # Exclude zero-variance pixels after candidate masking.

            arr = np.load(
                new_path,
                mmap_mode="r+",
                allow_pickle=False,
            )

            var_for_mean = np.load(
                var_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            mean = chunked_mean_and_subtract(
                arr,
                variance=var_for_mean,
                rows=rows,
            )

            del arr
            del var_for_mean

            old_work = state["work_file"]

            state["work_file"] = new_name

            state["clipped_for_level"] = True

            state["last_clip_mean"] = mean

            atomic_checkpoint_state(
                state_path,
                state,
            )

            print(
                f"\nThreshold {level+1}/{len(thresholds)}: "
                f"{threshold:g} — clipped; mean={mean:.6g}"
            )

            # Delete old file only after checkpoint commit.

            if old_work != new_name:

                try:

                    (
                        run_dir / old_work
                    ).unlink()

                except FileNotFoundError:

                    pass

            work_path = (
                run_dir / state["work_file"]
            )

            var_path = (
                run_dir / state["variance_file"]
            )

        # -----------------------------------------------------------
        # Process both orientations and repeated candidates
        # -----------------------------------------------------------

        while int(
            state["orientation_index"]
        ) < 2:

            ori = int(
                state["orientation_index"]
            )

            transpose = [
                False,
                True,
            ][ori]

            iteration = int(
                state["iteration_index"]
            )

            call_id = (
                f"t{level:02d}_o{ori}_i{iteration:02d}"
            )

            result_path = (
                run_dir / f"result_{call_id}.json"
            )

            output_work = (
                run_dir / f"work_after_{call_id}.npy"
            )

            output_var = (
                run_dir / f"variance_after_{call_id}.npy"
            )

            # Resource preflight before every worker.

            mem_before_child = require_memory_headroom(
                args,
                f"FRT call {int(state['frt_calls']) + 1}",
            )

            print(
                "  pre-call memory:",
                mem_before_child,
            )

            pending = {

                "call_id": call_id,

                "threshold_index": level,

                "threshold": threshold,

                "orientation_index": ori,

                "transpose": transpose,

                "iteration_index": iteration,

                "result_json": str(result_path),

            }

            state["pending_call"] = pending

            atomic_checkpoint_state(
                state_path,
                state,
            )

            # Remove stale outputs from previous failed attempt.

            for stale in (

                result_path,

                output_work,

                output_var,

                output_work.with_suffix(
                    output_work.suffix + ".tmp.npy"
                ),

                output_var.with_suffix(
                    output_var.suffix + ".tmp.npy"
                ),

            ):

                try:

                    stale.unlink()

                except FileNotFoundError:

                    pass

            cmd = [

                sys.executable,

                str(worker_script_path()),

                "--worker",

                "--repo",
                str(repo),

                "--commit",
                args.commit,

                "--source-sha256",
                source_tree_hash,

                "--work",
                str(work_path),

                "--variance",
                str(var_path),

                "--psf",
                str(psf_path),

                "--result-json",
                str(result_path),

                "--output-work",
                str(output_work),

                "--output-variance",
                str(output_var),

                "--threshold",
                str(threshold),

                "--transpose",
                "1" if transpose else "0",

                "--min-threshold",
                str(args.min_threshold),

                "--num-iterations",
                str(args.num_iterations),

                "--min-length",
                str(args.min_length),

                "--var-scalar",
                str(var_scalar),

                "--subtract-mean",
                "1",

            ]

            env = os.environ.copy()

            env.update({

                "OPENBLAS_NUM_THREADS": "1",

                "OMP_NUM_THREADS": "1",

                "MKL_NUM_THREADS": "1",

                "NUMEXPR_NUM_THREADS": "1",

                "MPLBACKEND": "Agg",

                "PYTHONUNBUFFERED": "1",

                "MALLOC_ARENA_MAX": "2",

                "MALLOC_TRIM_THRESHOLD_": "65536",

            })

            proc = subprocess.run(

                cmd,

                env=env,

                capture_output=True,

                text=True,

                check=False,

            )

            if proc.stdout:

                print(
                    proc.stdout.rstrip()
                )

            if (
                proc.returncode != 0
                and not result_path.exists()
            ):

                print(
                    proc.stderr[-8000:]
                    if proc.stderr
                    else "Child process exited without a result."
                )

                print(
                    "PAUSED: worker failed or was killed. "
                    "Checkpoint remains at pre-call image."
                )

                return 2

            if not result_path.exists():

                print(
                    proc.stderr[-8000:]
                    if proc.stderr
                    else "Child returned without result JSON."
                )

                raise RuntimeError(
                    "FRT child returned without a result file."
                )

            result = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )

            if result.get("status") == "worker_error":

                print(
                    result.get(
                        "traceback",
                        result.get(
                            "error",
                            "Unknown worker error",
                        ),
                    )
                )

                # Corrected recovery handling:
                # preserve the old image and clear pending worker result.

                result_path.unlink(
                    missing_ok=True
                )

                state["pending_call"] = None

                atomic_checkpoint_state(
                    state_path,
                    state,
                )

                print(
                    "PAUSED: worker error. "
                    "Checkpoint image was not advanced."
                )

                print(
                    "Fix the reported error, then rerun the same command."
                )

                return 2

            commit_child_result(
                pending,
                result,
            )

            calls_this_run += 1

            result_path.unlink(
                missing_ok=True
            )

            work_path = (
                run_dir / state["work_file"]
            )

            var_path = (
                run_dir / state["variance_file"]
            )

            if (
                memory_snapshot()["host_available_gib"]
                < args.pause_host_available_gib
            ):

                print(
                    "PAUSED after a complete FRT call "
                    "because host RAM headroom is low."
                )

                print(
                    "Current checkpoint:",
                    state_path,
                )

                return 0

            if (
                args.max_calls_per_run > 0
                and calls_this_run >= args.max_calls_per_run
            ):

                print(
                    f"PAUSED after requested limit "
                    f"of {calls_this_run} FRT call(s)."
                )

                print(
                    "This is a resumable checkpoint, "
                    "not a completed scan."
                )

                print(
                    "Rerun the same command without "
                    "--max-calls-per-run to continue."
                )

                print(
                    "Current checkpoint:",
                    state_path,
                )

                return 0

        # Both orientations complete:
        # advance to next clipping threshold.

        state["next_threshold_index"] = level + 1

        state["clipped_for_level"] = False

        state["orientation_index"] = 0

        state["iteration_index"] = 0

        atomic_checkpoint_state(
            state_path,
            state,
        )

    # ---------------------------------------------------------------
    # Complete scan
    # ---------------------------------------------------------------

    state["complete"] = True

    state["completed_unix"] = time.time()

    atomic_checkpoint_state(
        state_path,
        state,
    )

    save_csv()

    summary = {

        "frame": args.frame,

        "complete": True,

        "thresholds": thresholds,

        "minimum_radon_snr": float(
            args.min_threshold
        ),

        "clip_mode": args.clip_mode,

        "variance_mask_update_after_streak_removal": True,

        "iterations_per_orientation": int(
            args.num_iterations
        ),

        "frt_calls": int(
            state["frt_calls"]
        ),

        "candidate_count": len(
            state["candidates"]
        ),

        "max_pointwise_snr": float(
            max_snr
        ),

        "dynamic_range": float(
            dynamic_range
        ),

        "worker_process_per_frt_call": True,

        "memory_at_completion": memory_snapshot(),

        "elapsed_seconds_this_run": float(
            time.perf_counter() - start_time
        ),

        "candidate_csv": str(
            csv_path
        ),

        "state_json": str(
            state_path
        ),

        "science_notes": [

            "Candidate-generation threshold 7.5 is not a "
            "false-alarm-calibrated ILMT threshold.",

            "Pinned PyRaDoN long-mode compatibility patch "
            "is applied in each isolated worker.",

            "The exact Streak model footprint is set to zero "
            "in both image and variance, following Nir et al. "
            "2018 Section 4.4, footnote 6.",

            "clip_mode=pyradon_scalar reproduces the pinned "
            "package clipping cap.",

            "Full-frame mode is retained because sectioning "
            "could split the approximately 4553-pixel target.",

        ],

    }

    atomic_json(
        summary_path,
        summary,
    )

    print(
        "\nSTAGE 8.3F COMPLETE"
    )

    print(
        "FRT calls:",
        summary["frt_calls"],
    )

    print(
        "Candidates:",
        summary["candidate_count"],
    )

    print(
        "Candidate CSV:",
        csv_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "State:",
        state_path,
    )

    return 0


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        description=__doc__
    )

    p.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    p.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE),
    )

    p.add_argument(
        "--variance",
        default=str(DEFAULT_VARIANCE),
    )

    p.add_argument(
        "--psf",
        default=str(DEFAULT_PSF),
    )

    p.add_argument(
        "--repo",
        default=str(DEFAULT_REPO),
    )

    p.add_argument(
        "--run-dir",
        default=str(DEFAULT_RUN_DIR),
    )

    p.add_argument(
        "--commit",
        default=EXPECTED_COMMIT,
    )

    p.add_argument(
        "--source-sha256",
        default="",
        help=argparse.SUPPRESS,
    )

    p.add_argument(
        "--frame",
        default="20221026-0057.fits",
    )

    p.add_argument(
        "--min-threshold",
        type=float,
        default=7.5,
    )

    p.add_argument(
        "--num-iterations",
        type=int,
        default=3,
    )

    p.add_argument(
        "--min-length",
        type=int,
        default=32,
    )

    p.add_argument(
        "--var-scalar",
        type=float,
        default=303.41033935546875,
    )

    p.add_argument(
        "--clip-mode",
        choices=[
            "local_snr_cap",
            "pyradon_scalar",
        ],
        default="pyradon_scalar",
        help=(
            "Use pinned PyRaDoN scalar clipping cap "
            "for the baseline. local_snr_cap is a "
            "heteroskedastic sensitivity experiment."
        ),
    )

    p.add_argument(
        "--min-host-available-gib",
        type=float,
        default=7.0,
    )

    p.add_argument(
        "--min-cgroup-headroom-gib",
        type=float,
        default=7.0,
    )

    p.add_argument(
        "--pause-host-available-gib",
        type=float,
        default=2.5,
    )

    p.add_argument(
        "--max-calls-per-run",
        type=int,
        default=0,
    )

    p.add_argument(
        "--work",
        default="",
    )

    p.add_argument(
        "--result-json",
        default="",
    )

    p.add_argument(
        "--output-work",
        default="",
    )

    p.add_argument(
        "--output-variance",
        default="",
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=7.5,
    )

    p.add_argument(
        "--transpose",
        type=int,
        choices=[0, 1],
        default=0,
    )

    p.add_argument(
        "--subtract-mean",
        type=int,
        choices=[0, 1],
        default=1,
    )

    p.add_argument(
        "--psf-sigma-check",
        type=float,
        default=0.0,
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    args = build_parser().parse_args()

    if args.worker:

        required = [

            args.work,

            args.variance,

            args.psf,

            args.result_json,

            args.output_work,

            args.output_variance,

            args.repo,

        ]

        if any(not x for x in required):

            print(
                "Worker arguments incomplete.",
                file=sys.stderr,
            )

            return 2

        return child_worker(
            args
        )

    if (
        args.num_iterations < 1
        or args.min_length < 1
        or args.min_threshold <= 0
    ):

        raise ValueError(
            "Threshold and iteration/min-length values must be positive."
        )

    if args.max_calls_per_run < 0:

        raise ValueError(
            "--max-calls-per-run must be >= 0."
        )

    return scan_parent(
        args
    )


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
