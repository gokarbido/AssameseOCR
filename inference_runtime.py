"""Shared ONNX session setup for native Jetson CUDA inference."""

import onnxruntime as ort


def create_session(path, providers=None):
    """Require requested providers, including after ORT's startup fallback."""
    providers = list(providers) if providers else ['CUDAExecutionProvider', 'CPUExecutionProvider']
    available = ort.get_available_providers()
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise RuntimeError(
            f"Missing ONNX Runtime providers: {missing}; available: {available}. "
            "Install a Jetson GPU wheel matching Python, CUDA and cuDNN (see Readme.md), "
            "or explicitly use --device cpu."
        )
    session = ort.InferenceSession(str(path), providers=providers)
    active = session.get_providers()
    if any(provider not in active for provider in providers):
        raise RuntimeError(
            f"Requested {providers}, but model {path} activated {active}. "
            "Check the CUDA/cuDNN libraries and ONNX Runtime build; refusing CPU-only fallback."
        )
    session.disable_fallback()
    print(f"ONNX model: {path}; active providers: {active}")
    return session
