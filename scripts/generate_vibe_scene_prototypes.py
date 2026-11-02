from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main  # noqa: F401  # preload runtime dependencies
import open_clip

from grouping.vibe.cache import compute_model_fingerprint
from grouping.vibe.prototypes import (
    ACTION_CONCEPTS,
    MODEL_NAME,
    PRETRAINED_NAME,
    PROTOTYPE_DATA_FILENAME,
    PROTOTYPE_METADATA_FILENAME,
    PROTOTYPE_VERSION,
    PROMPT_TEMPLATES,
    SCENE_CONCEPTS,
    SHOT_TYPE_CONCEPTS,
    prompt_variants,
    prototype_sha256,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local CLIP scene prototype resources for vibe grouping.",
    )
    parser.add_argument(
        "--semantic-model",
        default=str(PROJECT_ROOT / "onnx_models" / "vibe_semantic_v1.onnx"),
        help="Path to the ONNX image encoder that the prototypes must match.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "onnx_models"),
        help="Directory where the .npz and .json resources will be written.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="OpenCLIP model name used to generate text embeddings.",
    )
    parser.add_argument(
        "--pretrained",
        default=PRETRAINED_NAME,
        help="OpenCLIP pretrained checkpoint identifier.",
    )
    return parser.parse_args()


def _encode_concepts(
    model: torch.nn.Module,
    tokenizer,
    concepts,
) -> np.ndarray:
    encoded: list[np.ndarray] = []
    with torch.inference_mode():
        for concept in concepts:
            tokens = tokenizer(prompt_variants(concept))
            embeddings = model.encode_text(tokens, normalize=True)
            mean_embedding = embeddings.mean(dim=0)
            mean_embedding = mean_embedding / mean_embedding.norm(p=2).clamp_min(1e-12)
            encoded.append(mean_embedding.cpu().numpy().astype(np.float32))
    return np.stack(encoded, axis=0)


def main_cli() -> None:
    args = _parse_args()
    semantic_model_path = Path(args.semantic_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, _, _ = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device="cpu",
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model_name)

    action_embeddings = _encode_concepts(model, tokenizer, ACTION_CONCEPTS)
    scene_embeddings = _encode_concepts(model, tokenizer, SCENE_CONCEPTS)
    shot_embeddings = _encode_concepts(model, tokenizer, SHOT_TYPE_CONCEPTS)

    data_path = output_dir / PROTOTYPE_DATA_FILENAME
    np.savez_compressed(
        data_path,
        action_embeddings=action_embeddings,
        scene_embeddings=scene_embeddings,
        shot_embeddings=shot_embeddings,
    )

    metadata = {
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "embedding_dimension": int(action_embeddings.shape[1]),
        "prototype_version": PROTOTYPE_VERSION,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "semantic_model_fingerprint": compute_model_fingerprint(semantic_model_path),
        "sha256": prototype_sha256(data_path),
        "action_concepts": [
            {
                "key": concept.key,
                "phrase": concept.phrase,
                "label": concept.label,
                "tags": list(concept.tags),
            }
            for concept in ACTION_CONCEPTS
        ],
        "scene_concepts": [
            {
                "key": concept.key,
                "phrase": concept.phrase,
                "label": concept.label,
                "tags": list(concept.tags),
            }
            for concept in SCENE_CONCEPTS
        ],
        "shot_concepts": [
            {
                "key": concept.key,
                "phrase": concept.phrase,
                "label": concept.label,
                "tags": list(concept.tags),
            }
            for concept in SHOT_TYPE_CONCEPTS
        ],
    }
    metadata_path = output_dir / PROTOTYPE_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {data_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main_cli()
