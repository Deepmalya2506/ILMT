# ============================================================================
# STAGE 8.3F-EXPORT — EXPORT MINIMAL IMMUTABLE INPUTS
#
# Run this cell AFTER Stage 8.2, while the notebook kernel is alive.
#
# No FRT calculation is performed here.
#
# Exports:
#   1. Stage 8.2 PSF-filtered image
#   2. Exact Stage 8.1 variance map attached to Finder
#   3. Empirical PSF
#   4. Provenance manifest
#   5. Terminal command for the isolated FRT runner
# ============================================================================

import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import numpy as np


_REQUIRED = [
    "finder",
    "FRT_INITIAL_WORK_IMAGE_PATH",
    "FRT_RUN_DIR",
    "FRAME_OUTPUT_DIR",
    "FRAME_FILENAME",
    "PYRADON_REPO",
    "PYRADON_COMMIT",
    "FRT_PSF_KERNEL",
    "frt_variance_input",
    "FRT_INPUT_SHA256",
]

_missing = [x for x in _REQUIRED if x not in globals()]

if _missing:
    raise RuntimeError(
        "Run through Stage 8.2 first. Missing: " + ", ".join(_missing)
    )


_output_dir = Path(FRAME_OUTPUT_DIR).resolve()
_output_dir.mkdir(parents=True, exist_ok=True)

_image_path = Path(FRT_INITIAL_WORK_IMAGE_PATH).resolve()

_variance_path = (
    _output_dir / "frt_empirical_variance_proxy.npy"
)

_psf_path = (
    _output_dir / "frt_empirical_psf_kernel.npy"
)

_manifest_path = (
    _output_dir / "frt_isolated_runner_inputs.json"
)


# ---------------------------------------------------------------------------
# 1. Validate immutable Stage 8.2 image
# ---------------------------------------------------------------------------

if not _image_path.is_file():
    raise FileNotFoundError(
        f"Stage 8.2 image does not exist: {_image_path}"
    )

_image = np.load(
    _image_path,
    mmap_mode="r",
    allow_pickle=False,
)

if _image.shape != (4096, 4096):
    raise RuntimeError(
        f"Unexpected image shape: {_image.shape}"
    )

if _image.dtype != np.float32:
    raise RuntimeError(
        f"Unexpected image dtype: {_image.dtype}"
    )


# ---------------------------------------------------------------------------
# 2. Retrieve exact variance map attached to Finder
# ---------------------------------------------------------------------------

_variance = finder.data.get_var_image(
    (0, 0),
    _image.shape,
)

if _variance.shape != _image.shape:
    raise RuntimeError(
        f"Finder variance map shape mismatch: {_variance.shape}"
    )

if np.asarray(_variance).dtype != np.float32:
    _variance = np.asarray(
        _variance,
        dtype=np.float32,
    )

if (
    not np.isfinite(_variance).all()
    or np.any(_variance <= 0)
):
    raise RuntimeError(
        "Initial Stage 8.1 variance map must be finite "
        "and strictly positive."
    )


# ---------------------------------------------------------------------------
# 3. Verify variance map against Stage 8.1 input
# ---------------------------------------------------------------------------

_live_variance = np.asarray(
    frt_variance_input,
    dtype=np.float32,
)

if _live_variance.shape != _variance.shape:
    raise RuntimeError(
        "frt_variance_input shape differs from Finder variance."
    )

for _r0 in range(0, _variance.shape[0], 256):

    _r1 = min(
        _r0 + 256,
        _variance.shape[0],
    )

    if not np.allclose(
        _variance[_r0:_r1],
        _live_variance[_r0:_r1],
        rtol=1e-6,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Finder variance differs from frt_variance_input "
            f"at rows {_r0}:{_r1}."
        )


# ---------------------------------------------------------------------------
# 4. Validate empirical PSF
# ---------------------------------------------------------------------------

_psf = np.asarray(
    FRT_PSF_KERNEL,
    dtype=np.float64,
)

if (
    _psf.ndim != 2
    or not np.isfinite(_psf).all()
    or np.any(_psf < 0)
):
    raise RuntimeError(
        "Empirical PSF must be a finite, non-negative 2-D array."
    )

if not np.isclose(
    float(_psf.sum()),
    1.0,
    rtol=1e-6,
    atol=1e-8,
):
    raise RuntimeError(
        f"Empirical PSF is not unit-normalized: sum={float(_psf.sum())}"
    )


# ---------------------------------------------------------------------------
# 5. Atomic NPY export
# ---------------------------------------------------------------------------

def _atomic_save_npy(
    path: Path,
    array: np.ndarray,
) -> None:

    tmp = path.with_name(path.name + ".tmp")

    with tmp.open("wb") as handle:

        np.save(
            handle,
            array,
            allow_pickle=False,
        )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        tmp,
        path,
    )


_atomic_save_npy(
    _variance_path,
    _variance,
)

_atomic_save_npy(
    _psf_path,
    _psf,
)


# ---------------------------------------------------------------------------
# 6. Validate scalar variance
# ---------------------------------------------------------------------------

_var_scalar = float(
    finder.data.var_scalar
)

if (
    not np.isfinite(_var_scalar)
    or _var_scalar <= 0
):
    raise RuntimeError(
        f"Invalid Finder var_scalar: {_var_scalar}"
    )


# ---------------------------------------------------------------------------
# 7. Write provenance manifest
# ---------------------------------------------------------------------------

_manifest = {

    "frame": str(FRAME_FILENAME),

    "image_path": str(_image_path),

    "image_shape": list(_image.shape),

    "image_dtype": str(_image.dtype),

    "variance_path": str(
        _variance_path.resolve()
    ),

    "variance_shape": list(_variance.shape),

    "variance_dtype": str(
        np.asarray(_variance).dtype
    ),

    "variance_source": (
        "finder.data.get_var_image((0,0), image.shape), "
        "checked against frt_variance_input"
    ),

    "psf_path": str(
        _psf_path.resolve()
    ),

    "psf_shape": list(_psf.shape),

    "psf_sum": float(_psf.sum()),

    "var_scalar": _var_scalar,

    "input_sha256": str(FRT_INPUT_SHA256),

    "pyradon_repo": str(
        Path(PYRADON_REPO).resolve()
    ),

    "pyradon_commit": str(PYRADON_COMMIT),

    "python_executable": sys.executable,

    "finder_psf_already_applied_to_image": True,

    "export_only_no_frt": True,

}

_manifest_path.write_text(
    json.dumps(_manifest, indent=2),
    encoding="utf-8",
)


# ---------------------------------------------------------------------------
# 8. Construct terminal command
# ---------------------------------------------------------------------------

_runner_path = (
    Path(PYRADON_REPO).resolve().parent
    / "ARIES_stage83F_isolated_frt_runner.py"
)

_run_dir = (
    Path(FRT_RUN_DIR).resolve()
    / "native_threshold_scan_isolated_v1"
)

_command = [

    sys.executable,

    str(_runner_path),

    "--image",
    str(_image_path),

    "--variance",
    str(_variance_path.resolve()),

    "--psf",
    str(_psf_path.resolve()),

    "--repo",
    str(Path(PYRADON_REPO).resolve()),

    "--commit",
    str(PYRADON_COMMIT),

    "--run-dir",
    str(_run_dir),

    "--var-scalar",
    str(_var_scalar),

    "--clip-mode",
    "pyradon_scalar",

    "--max-calls-per-run",
    "2",

]


# ---------------------------------------------------------------------------
# 9. Report
# ---------------------------------------------------------------------------

print(
    "STAGE 8.3F-EXPORT COMPLETE — NO FRT EXECUTED"
)

print(
    "------------------------------------------------"
)

print("Image:", _image_path)

print(
    "Variance:",
    _variance_path.resolve(),
)

print(
    "PSF:",
    _psf_path.resolve(),
)

print(
    "var_scalar:",
    _var_scalar,
)

print(
    "PyRaDoN repo:",
    Path(PYRADON_REPO).resolve(),
)

print(
    "PyRaDoN commit:",
    PYRADON_COMMIT,
)

print(
    "Python executable:",
    sys.executable,
)

print(
    "Manifest:",
    _manifest_path,
)

print(
    "\nPILOT COMMAND (two isolated FRT calls):"
)

print(
    " ".join(
        shlex.quote(str(x))
        for x in _command
    )
)

print(
    "\nAfter the two-call pilot succeeds, resume "
    "with the same command but remove:"
)

print(
    "  --max-calls-per-run 2"
)

print(
    "\nImportant: copy the runner .py file into "
    "the project root before running the command."
)