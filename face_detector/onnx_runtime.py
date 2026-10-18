from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
import sys
import threading
import time
from typing import Any, Sequence

from automatic_scan.perf import get_active_profiler
from automatic_scan.scan_logging import get_scan_logger
from .onnx_directml_compat import compatible_model_path

try:
    import onnxruntime as ort
except ImportError:
    ort = None

LOGGER = logging.getLogger(__name__)
Provider = str | tuple[str, dict[str, Any]]
CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"
_GPU_PROVIDERS = {"CUDAExecutionProvider", "MIGraphXExecutionProvider", DML,
                  "ROCMExecutionProvider"}
_OVERRIDE_PROVIDERS = {
    "cuda": "CUDAExecutionProvider", "migraphx": "MIGraphXExecutionProvider",
    "directml": DML, "cpu": CPU,
}
_PRELOAD_ATTEMPTED = False
_STARTUP_LOGGED = False


def _resolve_runtime_attribute(name: str) -> Any:
    if ort is None:
        return None
    candidate = getattr(ort, name, None)
    if candidate is not None:
        return candidate
    capi = getattr(ort, "capi", None)
    for state_name in ("_pybind_state", "onnxruntime_pybind11_state"):
        candidate = getattr(getattr(capi, state_name, None), name, None)
        if candidate is not None:
            return candidate
    return None


def get_available_providers() -> list[str]:
    # get_all_providers lists compiled provider names, not usable providers.
    loader = _resolve_runtime_attribute("get_available_providers")
    if callable(loader):
        try:
            return list(dict.fromkeys(str(p) for p in loader()))
        except Exception:
            LOGGER.warning("Could not query ONNX Runtime providers; using CPU.")
    return [CPU]


def auto_provider_order(platform: str | None = None) -> tuple[str, ...]:
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return ("CUDAExecutionProvider", "MIGraphXExecutionProvider", DML,
                "OpenVINOExecutionProvider", CPU)
    if platform == "darwin":
        return ("CoreMLExecutionProvider", CPU)
    # ROCm is only a compatibility candidate in older Linux runtimes.
    return ("CUDAExecutionProvider", "MIGraphXExecutionProvider",
            "OpenVINOExecutionProvider", "ROCMExecutionProvider", CPU)


DEFAULT_PREFERRED_PROVIDERS = auto_provider_order()


def _provider_name(provider: Provider) -> str:
    return provider[0] if isinstance(provider, tuple) else provider


def provider_device(provider: str | None) -> str:
    """Classify the primary EP; mixed-provider graphs can still run CPU nodes."""
    if provider == CPU:
        return "CPU"
    if provider in _GPU_PROVIDERS:
        return "GPU"
    return "UNKNOWN"


def select_providers(
    preferred_providers: Sequence[Provider] | None = None,
    *, platform: str | None = None,
) -> list[Provider]:
    available = set(get_available_providers())
    if preferred_providers is not None:
        requested = list(preferred_providers)
    else:
        override = os.environ.get("ROOM36_ONNX_PROVIDER", "auto").strip().lower()
        if override == "cpu":
            return [CPU]
        requested = list(auto_provider_order(platform))
        if override in _OVERRIDE_PROVIDERS:
            requested.insert(0, _OVERRIDE_PROVIDERS[override])
        elif override not in ("auto", ""):
            LOGGER.warning("Unsupported ROOM36_ONNX_PROVIDER=%r; using AUTO.", override)
    selected: list[Provider] = []
    names: set[str] = set()
    for provider in requested:
        name = _provider_name(provider)
        if name in available and name not in names:
            selected.append(provider)
            names.add(name)
    if CPU not in names:
        selected.append(CPU)
    return selected


def preload_onnxruntime_dependencies() -> None:
    """Optional CUDA preload. Never required by DirectML or CPU."""
    global _PRELOAD_ATTEMPTED
    if _PRELOAD_ATTEMPTED or "CUDAExecutionProvider" not in get_available_providers():
        return
    _PRELOAD_ATTEMPTED = True
    preload = _resolve_runtime_attribute("preload_dlls")
    if callable(preload):
        try:
            preload(cuda=True, cudnn=True, msvc=True)
        except Exception:
            LOGGER.warning("Optional CUDA preload failed; other providers remain available.")


def startup_diagnostics() -> dict[str, Any]:
    available = get_available_providers()
    selected = select_providers()
    return {
        "runtime_version": getattr(ort, "__version__", "unavailable"),
        "requested_provider": os.environ.get("ROOM36_ONNX_PROVIDER", "auto"),
        "available_providers": available,
        "selected_providers": [_provider_name(p) for p in selected],
    }


def log_startup_diagnostics() -> None:
    global _STARTUP_LOGGED
    if _STARTUP_LOGGED:
        return
    _STARTUP_LOGGED = True
    info = startup_diagnostics()
    LOGGER.info("ONNX Runtime providers available: %s; selected inference provider: %s",
                ", ".join(info["available_providers"]), info["selected_providers"][0])


def _configure_options(options: Any, providers: Sequence[Provider]) -> None:
    # Each attempt has one accelerator and CPU; only a primary DML attempt needs this.
    if providers and _provider_name(providers[0]) == DML:
        execution_mode = _resolve_runtime_attribute("ExecutionMode")
        options.execution_mode = execution_mode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False


def create_session_options(providers: Sequence[Provider] | None = None) -> Any:
    options_type = _resolve_runtime_attribute("SessionOptions")
    if options_type is None:
        raise RuntimeError("A compatible ONNX Runtime package is required.")
    options = options_type()
    optimization = _resolve_runtime_attribute("GraphOptimizationLevel")
    if optimization is not None:
        options.graph_optimization_level = optimization.ORT_ENABLE_ALL
    options.log_severity_level = 3
    _configure_options(options, select_providers() if providers is None else providers)
    return options


class InferenceSession:
    """Delegate ORT API; serialize Run only for this session when DML is active."""

    def __init__(self, session: Any, diagnostics: dict[str, Any], *,
                 model_path: str | Path | None = None,
                 providers: Sequence[Provider] = (),
                 options: Any | None = None,
                 batch_variant: bool = False) -> None:
        self._session = session
        self.diagnostics = diagnostics
        self._scan_logger = get_scan_logger()
        self._model_name = diagnostics.get("model") or (Path(model_path).stem if model_path else "unknown")
        self._run_lock = threading.RLock() if DML in diagnostics["active_providers"] else None
        self._model_path = model_path
        self._providers = providers
        self._options = options
        self._batch_sessions: OrderedDict[int, InferenceSession] = OrderedDict()
        self._batch_symbol = None
        if self._run_lock is not None and not batch_variant and callable(getattr(session, "get_inputs", None)):
            shape = session.get_inputs()[0].shape
            # An older ArcFace export uses the literal symbol "None" for its
            # free batch axis. ORT DirectML needs that axis fixed for batch >1.
            if shape and shape[0] == "None":
                self._batch_symbol = shape[0]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def get_providers(self) -> list[str]:
        return self.diagnostics.get("last_run_providers", self._session.get_providers())

    def _execute_inference(self, operation: str, call, *args: Any, **kwargs: Any) -> Any:
        providers = list(self._session.get_providers())
        primary = providers[0] if providers else None
        feed = args[1] if operation == "run" and len(args) > 1 else kwargs.get("input_feed")
        values = list(feed.values()) if isinstance(feed, dict) else []
        shape = getattr(values[0], "shape", ()) if values else ()
        batch = str(shape[0]) if shape else "unknown"
        started = time.perf_counter()
        try:
            result = call(*args, **kwargs)
        except Exception:
            self._scan_logger.exception(
                "onnx_inference_failed model=%s operation=%s provider=%s device=%s providers=%s batch_size=%s",
                self._model_name, operation, primary, provider_device(primary),
                ",".join(providers), batch)
            raise
        self.diagnostics["last_run_providers"] = providers
        self._scan_logger.info(
            "onnx_inference model=%s operation=%s provider=%s device=%s providers=%s batch_size=%s seconds=%.6f",
            self._model_name, operation, primary, provider_device(primary),
            ",".join(providers), batch, time.perf_counter() - started)
        return result

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if self._run_lock is None:
            return self._execute_inference("run", self._session.run, *args, **kwargs)
        with self._run_lock:
            feeds = args[1] if len(args) > 1 else kwargs.get("input_feed", {})
            for value in feeds.values():
                if any(size == 0 for size in getattr(value, "shape", ())):
                    raise ValueError("DirectML does not accept empty input dimensions.")
            if self._batch_symbol and self._model_path and feeds:
                batch = int(next(iter(feeds.values())).shape[0])
                if batch > 1:
                    variant = self._batch_sessions.get(batch)
                    if variant is None:
                        options = _copy_session_options(self._options)
                        options.add_free_dimension_override_by_name(self._batch_symbol, batch)
                        variant = create_inference_session(
                            self._model_path, providers=self._providers,
                            session_options=options, _batch_variant=True)
                        self._batch_sessions[batch] = variant
                        if len(self._batch_sessions) > 2:
                            self._batch_sessions.popitem(last=False)
                    else:
                        self._batch_sessions.move_to_end(batch)
                    outputs = variant.run(*args, **kwargs)
                    self.diagnostics["last_run_providers"] = variant.get_providers()
                    self.diagnostics["last_run_fallbacks"] = variant.diagnostics["fallbacks"]
                    return outputs
            return self._execute_inference("run", self._session.run, *args, **kwargs)

    def run_with_iobinding(self, *args: Any, **kwargs: Any) -> Any:
        if self._run_lock is None:
            return self._execute_inference("run_with_iobinding", self._session.run_with_iobinding,
                                           *args, **kwargs)
        with self._run_lock:
            return self._execute_inference("run_with_iobinding", self._session.run_with_iobinding,
                                           *args, **kwargs)

    def run_async(self, *args: Any, **kwargs: Any) -> Any:
        if self._run_lock is not None:
            raise RuntimeError("Use synchronous Run on a background worker for DirectML.")
        providers = list(self._session.get_providers())
        primary = providers[0] if providers else None
        result = self._session.run_async(*args, **kwargs)
        self._scan_logger.info("onnx_inference_submitted model=%s operation=run_async provider=%s device=%s providers=%s",
                               self._model_name, primary, provider_device(primary), ",".join(providers))
        return result


def _copy_session_options(source: Any | None, providers: Sequence[Provider] = (DML, CPU)) -> Any:
    options = create_session_options(providers)
    if source is not None:
        for name in ("intra_op_num_threads", "inter_op_num_threads", "graph_optimization_level",
                     "log_severity_level", "enable_cpu_mem_arena"):
            try:
                setattr(options, name, getattr(source, name))
            except (AttributeError, TypeError):
                pass
    return options


def create_inference_session(
    model_path: str | Path,
    *, session_options: Any | None = None,
    providers: Sequence[Provider] | None = None,
    _batch_variant: bool = False,
) -> InferenceSession:
    if ort is None:
        raise RuntimeError("A compatible ONNX Runtime package is required.")
    log_startup_diagnostics()
    selected = select_providers(providers)
    available = get_available_providers()
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    last_error: Exception | None = None
    for index in range(len(selected)):
        primary = _provider_name(selected[index])
        if primary == "CUDAExecutionProvider":
            preload_onnxruntime_dependencies()
        candidates = [selected[index]] if primary == CPU else [selected[index], CPU]
        options = (session_options if index == 0 else _copy_session_options(session_options, candidates))
        if options is None:
            options = create_session_options(candidates)
        _configure_options(options, candidates)
        try:
            with compatible_model_path(model_path, directml=primary == DML) as (source, corrections):
                raw = ort.InferenceSession(source, sess_options=options, providers=candidates)
            active = list(raw.get_providers())
            # ORT may silently retry CPU inside its constructor. Give the next
            # accelerator a chance rather than accepting that premature fallback.
            if primary != CPU and (not active or active[0] != primary):
                failures.append({"provider": primary, "reason": "runtime_selected_other_provider"})
                get_scan_logger().warning("onnx_session_fallback model=%s requested=%s actual=%s reason=runtime_selected_other_provider",
                                          Path(model_path).stem, primary, active[0] if active else "none")
                LOGGER.warning("ONNX model %s: %s did not initialize; trying next provider.",
                               Path(model_path).stem, primary)
                del raw
                continue
            seconds = time.perf_counter() - started
            diagnostics = {
                "model": Path(model_path).stem,
                "requested_provider": _provider_name(selected[0]),
                "requested_providers": [_provider_name(p) for p in selected],
                "available_providers": available,
                "active_provider": active[0] if active else None,
                "active_providers": active,
                "fallbacks": failures,
                "directml_reshape_corrections": corrections,
                "session_creation_seconds": seconds,
            }
            profiler = get_active_profiler()
            if profiler is not None:
                profiler.record_model_session_creation(
                    Path(model_path).stem, seconds, provider=diagnostics["active_provider"])
            get_scan_logger().info("onnx_session_ready model=%s requested=%s active=%s device=%s providers=%s fallback_count=%d seconds=%.4f",
                                   diagnostics["model"], diagnostics["requested_provider"],
                                   diagnostics["active_provider"], provider_device(diagnostics["active_provider"]),
                                   ",".join(active), len(failures), seconds)
            return InferenceSession(raw, diagnostics, model_path=model_path,
                                    providers=candidates, options=options,
                                    batch_variant=_batch_variant)
        except Exception as exc:
            last_error = exc
            failures.append({"provider": primary, "reason": type(exc).__name__})
            get_scan_logger().exception("onnx_session_fallback model=%s provider=%s device=%s reason=%s",
                                        Path(model_path).stem, primary, provider_device(primary), type(exc).__name__)
            LOGGER.warning("ONNX model %s: %s initialization failed (%s); trying fallback.",
                           Path(model_path).stem, primary, type(exc).__name__)
    raise RuntimeError(f"Unable to initialize ONNX model {Path(model_path).stem}.") from last_error
