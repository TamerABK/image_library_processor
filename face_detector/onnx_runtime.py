from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import onnxruntime as ort


DEFAULT_PREFERRED_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "ROCMExecutionProvider",
    "CPUExecutionProvider",
)


def _resolve_runtime_attribute(name: str) -> Any:
    candidate = getattr(ort, name, None)
    if candidate is not None:
        return candidate

    capi = getattr(ort, "capi", None)
    if capi is None:
        return None

    for state_name in ("_pybind_state", "onnxruntime_pybind11_state"):
        state = getattr(capi, state_name, None)
        if state is None:
            continue

        candidate = getattr(state, name, None)
        if candidate is not None:
            return candidate

    return None


def _resolve_runtime_callable(name: str):
    candidate = _resolve_runtime_attribute(name)
    if callable(candidate):
        return candidate

    return None


def get_available_providers() -> list[str]:
    for attribute_name in ("get_available_providers", "get_all_providers"):
        provider_loader = _resolve_runtime_callable(attribute_name)
        if provider_loader is None:
            continue

        try:
            providers = provider_loader()
        except Exception:
            continue

        if providers:
            return [str(provider) for provider in providers]

    return ["CPUExecutionProvider"]


def select_providers(
    preferred_providers: Sequence[str] = DEFAULT_PREFERRED_PROVIDERS,
) -> list[str]:
    available = set(get_available_providers())
    selected = [
        provider
        for provider in preferred_providers
        if provider in available
    ]
    return selected or ["CPUExecutionProvider"]


def create_session_options() -> Any:
    session_options_type = _resolve_runtime_attribute("SessionOptions")
    if session_options_type is None:
        raise AttributeError(
            "onnxruntime.SessionOptions is unavailable. "
            "Python likely imported an incomplete ONNX Runtime package."
        )

    session_options = session_options_type()
    graph_optimization_level = getattr(ort, "GraphOptimizationLevel", None)
    if graph_optimization_level is not None:
        ort_enable_all = getattr(graph_optimization_level, "ORT_ENABLE_ALL", None)
        if ort_enable_all is not None:
            session_options.graph_optimization_level = ort_enable_all

    session_options.log_severity_level = 3
    return session_options


def create_inference_session(
    model_path: str | Path,
    *,
    session_options: Any | None = None,
    providers: Sequence[str] | None = None,
) -> ort.InferenceSession:
    session_options = session_options or create_session_options()

    provider_candidates: list[list[str] | None] = []
    if providers is not None:
        provider_candidates.append(list(providers))
    else:
        provider_candidates.append(select_providers())

    provider_candidates.extend(
        [
            ["CPUExecutionProvider"],
            None,
        ]
    )

    attempted: set[tuple[str, ...] | None] = set()
    last_error: Exception | None = None

    for candidate in provider_candidates:
        candidate_key = None if candidate is None else tuple(candidate)
        if candidate_key in attempted:
            continue
        attempted.add(candidate_key)

        try:
            if candidate is None:
                return ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options,
                )

            return ort.InferenceSession(
                str(model_path),
                sess_options=session_options,
                providers=candidate,
            )
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Unable to create ONNX Runtime session for {model_path}.")
