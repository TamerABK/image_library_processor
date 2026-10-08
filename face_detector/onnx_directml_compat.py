"""Narrow ONNX Reshape compatibility fix for DirectML.

The shipped pose and Vibe exports use allowzero=1 even where their shape
vectors contain only nonzero dimensions. DirectML 1.24.4 creates those
sessions, then fails at Run. For nonempty input tensors, allowzero=0 is
mathematically identical on these nodes. Validate that property before
changing only the affected attribute in an ephemeral model copy.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile


def _mentions_allowzero(path: Path) -> bool:
    previous = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if b"allowzero" in previous + chunk:
                return True
            previous = chunk[-8:]
    return False


def _nonzero_shape(name, *, constants, producers, graph_inputs, seen=None) -> bool:
    import numpy as np

    if name in constants:
        return bool(np.all(constants[name] != 0))
    seen = set() if seen is None else seen
    if name in seen:
        return False
    seen.add(name)
    node = producers.get(name)
    if node is None:
        return False
    if node.op_type == "Shape" and node.input[0] in graph_inputs:
        # Production model inputs have a positive batch and fixed positive
        # channel/height/width. InferenceSession rejects empty dimensions.
        return True
    if node.op_type == "Concat":
        return all(_nonzero_shape(part, constants=constants, producers=producers,
                                  graph_inputs=graph_inputs, seen=seen.copy())
                   for part in node.input)
    return False


def correct_nonzero_reshapes(model) -> int:
    from onnx import numpy_helper

    producers = {name: node for node in model.graph.node for name in node.output}
    needed = {node.input[1] for node in model.graph.node
              if node.op_type == "Reshape" and len(node.input) > 1}
    pending = list(needed)
    while pending:
        node = producers.get(pending.pop())
        if node is not None and node.op_type == "Concat":
            for value in node.input:
                if value not in needed:
                    needed.add(value)
                    pending.append(value)
    constants = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer
                 if item.name in needed}
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] in needed:
            for attribute in node.attribute:
                if attribute.name == "value":
                    constants[node.output[0]] = numpy_helper.to_array(attribute.t)
    graph_inputs = {value.name for value in model.graph.input}
    corrected = 0
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        if not _nonzero_shape(node.input[1], constants=constants, producers=producers,
                              graph_inputs=graph_inputs):
            continue
        for attribute in node.attribute:
            if attribute.name == "allowzero" and attribute.i == 1:
                attribute.i = 0
                corrected += 1
    return corrected


@contextmanager
def compatible_model_path(model_path: str | Path, *, directml: bool):
    original = Path(model_path)
    if not directml or not original.is_file() or not _mentions_allowzero(original):
        yield str(original), 0
        return
    import onnx

    model = onnx.load(original, load_external_data=True)
    corrected = correct_nonzero_reshapes(model)
    if not corrected:
        yield str(original), 0
        return
    with tempfile.TemporaryDirectory(prefix="room36-onnx-dml-") as directory:
        target = Path(directory) / original.name
        onnx.save_model(model, target, save_as_external_data=True,
                        all_tensors_to_one_file=True, location="weights.data",
                        size_threshold=1024)
        del model
        yield str(target), corrected
