"""Shared ONNX device selection for image, point-selection and video inference."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import onnxruntime as ort


LOGGER = logging.getLogger(__name__)
DEVICE_ENV = "AUTO_MOSAIC_DEVICE"


@lru_cache(maxsize=1)
def _preload_cuda() -> None:
    # Prefer this environment's NVIDIA wheels over unrelated PyTorch installs.
    ort.preload_dlls(directory="")


def create_session(model_path: Path) -> ort.InferenceSession:
    """Auto prefers CUDA; explicit cuda must succeed (useful for diagnostics)."""
    device = os.environ.get(DEVICE_ENV, "auto").lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"{DEVICE_ENV} must be auto, cpu or cuda: {device!r}")
    options = ort.SessionOptions()
    options.log_severity_level = 3
    use_cuda = device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA is unavailable. Install requirements-gpu.txt first.")
    providers = ["CPUExecutionProvider"]
    if use_cuda:
        try:
            _preload_cuda()
            # Keep FP32 precision: masks can change at threshold boundaries with TF32.
            providers.insert(0, ("CUDAExecutionProvider", {"use_tf32": "0"}))
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("ONNX Runtime could not initialize CUDA; check driver and CUDA DLLs.")
        except Exception:
            if device == "cuda":
                raise
            LOGGER.warning("CUDA initialization failed for %s; using CPU", model_path.name, exc_info=True)
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
            )
    else:
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    if device == "cuda":
        session.disable_fallback()
    LOGGER.info("%s: %s", model_path.name, session.get_providers())
    return session
