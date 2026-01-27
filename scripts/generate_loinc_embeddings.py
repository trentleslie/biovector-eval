#!/usr/bin/env python
"""Generate embeddings for LOINC questionnaire codes.

This script generates embeddings for LOINC codes using multiple embedding
models and strategies (single-primary, single-pooled, multi-vector).

Run overnight with:
    mkdir -p logs
    nohup uv run python scripts/generate_loinc_embeddings.py \
        --output-dir data/questionnaires/embeddings/loinc \
        --device cuda \
        --models all \
        --strategies all \
        > logs/loinc_embeddings_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from biovector_eval.base.entity import BaseEntity
from biovector_eval.core.embedding_strategy import (
    MultiVectorStrategy,
    StrategyType,
    get_strategy,
)
from biovector_eval.domains import get_domain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""

    name: str
    model_id: str
    dimension: int


# Models optimized for clinical/biomedical text
MODELS: dict[str, ModelConfig] = {
    "minilm": ModelConfig(
        name="MiniLM-L6-v2",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
    ),
    "bge-small": ModelConfig(
        name="BGE-Small",
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
    ),
    "sapbert": ModelConfig(
        name="SapBERT",
        model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        dimension=768,
    ),
    "pubmedbert": ModelConfig(
        name="PubMedBERT",
        model_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        dimension=768,
    ),
    "bge-m3": ModelConfig(
        name="BGE-M3",
        model_id="BAAI/bge-m3",
        dimension=1024,
    ),
}

# All available strategies
ALL_STRATEGIES = ["single-primary", "single-pooled", "multi-vector"]


def clear_gpu_memory() -> None:
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_slug(model_id: str) -> str:
    """Convert model ID to filesystem-safe slug."""
    return model_id.split("/")[-1].lower()


def entities_to_dicts(entities: list[BaseEntity]) -> list[dict[str, Any]]:
    """Convert BaseEntity objects to dicts for embedding generation.

    The embedding functions expect dicts with specific keys. We map:
    - id -> entity_id (to match metabolite format with hmdb_id)
    - name -> name
    - synonyms -> synonyms

    Args:
        entities: List of BaseEntity objects.

    Returns:
        List of dicts suitable for embedding generation.
    """
    return [
        {
            "entity_id": e.id,  # LOINC_NUM
            "name": e.name,  # LONG_COMMON_NAME
            "synonyms": e.synonyms,  # SHORTNAME, COMPONENT, consumer name
            "metadata": e.metadata,
        }
        for e in entities
    ]


def generate_single_primary(
    entities: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 512,
) -> tuple[np.ndarray, None]:
    """Generate one embedding per entity using primary name only.

    Args:
        entities: List of entity dicts with 'name' field.
        model: SentenceTransformer model.
        batch_size: Batch size for encoding.

    Returns:
        Tuple of (embeddings array, None for metadata).
    """
    texts = [e["name"] for e in entities]

    logger.info(f"Encoding {len(texts)} primary names...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype(np.float32), None


def generate_single_pooled(
    entities: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 512,
    max_synonyms: int = 15,
) -> tuple[np.ndarray, None]:
    """Generate one embedding per entity by averaging name + synonyms.

    Args:
        entities: List of entity dicts with 'name' and 'synonyms' fields.
        model: SentenceTransformer model.
        batch_size: Batch size for encoding.
        max_synonyms: Maximum synonyms to include per entity.

    Returns:
        Tuple of (embeddings array, None for metadata).
    """
    # Process in batches to avoid OOM
    entity_batch_size = 5000
    encode_batch_size = min(batch_size, 64)
    num_batches = (len(entities) + entity_batch_size - 1) // entity_batch_size

    logger.info(
        f"Processing {len(entities)} entities in {num_batches} batches "
        f"(encode_batch_size={encode_batch_size})..."
    )

    # Get embedding dimension
    test_emb = model.encode(["test"], convert_to_numpy=True)
    embed_dim = test_emb.shape[1]
    del test_emb
    clear_gpu_memory()

    # Pre-allocate output
    pooled_embeddings = np.zeros((len(entities), embed_dim), dtype=np.float32)

    for batch_idx in range(num_batches):
        start_e = batch_idx * entity_batch_size
        end_e = min(start_e + entity_batch_size, len(entities))
        batch_entities = entities[start_e:end_e]

        # Collect texts for this batch
        batch_texts: list[str] = []
        batch_ranges: list[tuple[int, int]] = []

        for e in batch_entities:
            text_start = len(batch_texts)
            texts_for_e = [e["name"]]
            synonyms = e.get("synonyms", [])[:max_synonyms]
            texts_for_e.extend(synonyms)
            batch_texts.extend(texts_for_e)
            batch_ranges.append((text_start, len(batch_texts)))

        logger.info(
            f"  Batch {batch_idx + 1}/{num_batches}: "
            f"{len(batch_entities)} entities, {len(batch_texts)} texts"
        )

        # Encode
        batch_embeddings = model.encode(
            batch_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )

        # Average per entity
        for i, (text_start, text_end) in enumerate(batch_ranges):
            e_embeddings = batch_embeddings[text_start:text_end]
            mean_emb = np.mean(e_embeddings, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            pooled_embeddings[start_e + i] = mean_emb

        del batch_embeddings, batch_texts, batch_ranges
        clear_gpu_memory()

    logger.info(f"Generated {len(pooled_embeddings)} pooled embeddings")
    return pooled_embeddings, None


def generate_multi_vector(
    entities: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int = 512,
    max_synonyms: int = 15,
    max_variations: int = 20,
) -> tuple[np.ndarray, dict[str, dict[str, Any]]]:
    """Generate multiple embeddings per entity with tier weights.

    Creates embeddings for:
    - Primary name (weight 1.0)
    - Synonyms (weight 0.9)
    - Generated variations (weight 0.7)

    Args:
        entities: List of entity dicts.
        model: SentenceTransformer model.
        batch_size: Batch size for encoding.
        max_synonyms: Maximum synonyms per entity.
        max_variations: Maximum variations per entity.

    Returns:
        Tuple of (embeddings array, metadata dict).
    """
    strategy = MultiVectorStrategy()

    # Phase 1: Collect all texts and metadata
    all_texts: list[str] = []
    all_metadata: list[dict[str, Any]] = []

    logger.info("Preparing multi-vector texts...")
    for e in entities:
        entity_id = e["entity_id"]
        name = e["name"]
        synonyms = e.get("synonyms", [])[:max_synonyms]

        # Primary name
        all_texts.append(name)
        all_metadata.append(
            {"entity_id": entity_id, "tier": "primary", "weight": 1.0, "text": name}
        )

        # Synonyms
        seen_lower = {name.lower()}
        for syn in synonyms:
            if syn.lower() not in seen_lower:
                all_texts.append(syn)
                all_metadata.append(
                    {"entity_id": entity_id, "tier": "synonym", "weight": 0.9, "text": syn}
                )
                seen_lower.add(syn.lower())

        # Variations
        variations = strategy._generate_variations(name, synonyms)
        for var in variations[:max_variations]:
            if var.lower() not in seen_lower:
                all_texts.append(var)
                all_metadata.append(
                    {
                        "entity_id": entity_id,
                        "tier": "variation",
                        "weight": 0.7,
                        "text": var,
                    }
                )
                seen_lower.add(var.lower())

    logger.info(f"Collected {len(all_texts)} texts from {len(entities)} entities")

    # Phase 2: Encode in chunks
    chunk_size = 20000
    encode_batch_size = min(batch_size, 32)
    num_chunks = (len(all_texts) + chunk_size - 1) // chunk_size
    logger.info(f"Encoding texts in {num_chunks} chunks (batch_size={encode_batch_size})...")

    # Get embedding dimension
    test_emb = model.encode(["test"], convert_to_numpy=True)
    embed_dim = test_emb.shape[1]
    del test_emb
    clear_gpu_memory()

    # Pre-allocate output
    all_embeddings = np.zeros((len(all_texts), embed_dim), dtype=np.float32)

    for i in range(0, len(all_texts), chunk_size):
        chunk_texts = all_texts[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        logger.info(f"  Chunk {chunk_num}/{num_chunks}: {len(chunk_texts)} texts")

        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        all_embeddings[i : i + len(chunk_texts)] = chunk_embeddings

        del chunk_embeddings
        clear_gpu_memory()

    # Phase 3: Build metadata mapping
    metadata = {str(i): meta for i, meta in enumerate(all_metadata)}

    logger.info(f"Generated {len(all_embeddings)} multi-vector embeddings")
    return all_embeddings.astype(np.float32), metadata


def process_model_strategy(
    config: ModelConfig,
    entities: list[dict[str, Any]],
    output_dir: Path,
    device: str,
    strategy_name: str,
    batch_size: int = 512,
) -> dict:
    """Process a model + strategy combination.

    Args:
        config: Model configuration.
        entities: List of entity dicts.
        output_dir: Output directory for embeddings.
        device: Device for inference (cuda/cpu).
        strategy_name: Strategy name.
        batch_size: Batch size for encoding.

    Returns:
        Result dict with timing and path information.
    """
    model_slug = get_model_slug(config.model_id)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {config.name} ({config.model_id})")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"{'='*60}")

    total_start = time.time()
    result = {
        "model": config.name,
        "model_id": config.model_id,
        "model_slug": model_slug,
        "strategy": strategy_name,
    }

    # Load model
    logger.info(f"Loading model on {device}...")
    model_start = time.time()
    model = SentenceTransformer(config.model_id, device=device)
    result["model_load_time"] = time.time() - model_start
    logger.info(f"Model loaded in {result['model_load_time']:.1f}s")

    # Generate embeddings based on strategy
    embed_start = time.time()
    if strategy_name == "single-primary":
        embeddings, metadata = generate_single_primary(entities, model, batch_size)
    elif strategy_name == "single-pooled":
        embeddings, metadata = generate_single_pooled(entities, model, batch_size)
    else:  # multi-vector
        embeddings, metadata = generate_multi_vector(entities, model, batch_size)

    result["embedding_time"] = time.time() - embed_start
    result["embedding_shape"] = list(embeddings.shape)
    result["num_vectors"] = embeddings.shape[0]

    logger.info(f"Generated {embeddings.shape[0]} vectors ({embeddings.shape[1]}d)")

    # Save embeddings
    output_dir.mkdir(parents=True, exist_ok=True)

    if strategy_name == "multi-vector":
        # Save to directory structure
        mv_dir = output_dir / f"{model_slug}_{strategy_name}"
        mv_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = mv_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)
        result["embeddings_path"] = str(embeddings_path)

        metadata_path = mv_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        result["metadata_path"] = str(metadata_path)

        logger.info(f"Saved embeddings: {embeddings_path}")
        logger.info(f"Saved metadata: {metadata_path}")
    else:
        # Save flat file
        embeddings_path = output_dir / f"{model_slug}_{strategy_name}.npy"
        np.save(embeddings_path, embeddings)
        result["embeddings_path"] = str(embeddings_path)
        logger.info(f"Saved embeddings: {embeddings_path}")

    result["embeddings_size_mb"] = embeddings.nbytes / 1024 / 1024

    # Cleanup
    del model, embeddings
    if metadata:
        del metadata
    clear_gpu_memory()

    result["total_time"] = time.time() - total_start
    logger.info(
        f"Completed {config.name} ({strategy_name}) in {result['total_time']:.1f}s"
    )

    return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for LOINC questionnaire codes"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model names or 'all'",
    )
    parser.add_argument(
        "--strategies",
        default="single-primary",
        help="Comma-separated strategies or 'all'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-parse LOINC CSV even if processed file exists",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Verify CUDA works if requested
    if device == "cuda":
        try:
            torch.cuda.init()
            logger.info(f"Using device: {device}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        except RuntimeError as e:
            logger.warning(f"CUDA initialization failed: {e}")
            logger.warning("Falling back to CPU.")
            device = "cpu"
    else:
        logger.info(f"Using device: {device}")

    # Load LOINC entities via domain registry
    logger.info("Loading LOINC entities...")
    loinc_domain = get_domain("loinc")

    if args.refresh:
        logger.info("Forcing re-parse of LOINC CSV...")
        base_entities = loinc_domain.refresh_entities()
    else:
        base_entities = loinc_domain.load_entities()

    logger.info(f"Loaded {len(base_entities)} LOINC codes")

    # Convert to dicts for embedding generation
    entities = entities_to_dicts(base_entities)

    # Save ID mapping
    id_mapping_path = output_dir / "id_mapping.json"
    id_mapping = {
        "ids": [e["entity_id"] for e in entities],
        "count": len(entities),
        "domain": "loinc",
    }
    with open(id_mapping_path, "w") as f:
        json.dump(id_mapping, f, indent=2)
    logger.info(f"Saved ID mapping: {id_mapping_path}")

    # Select models
    if args.models == "all":
        model_configs = list(MODELS.values())
    else:
        model_names = [m.strip() for m in args.models.split(",")]
        model_configs = [MODELS[m] for m in model_names if m in MODELS]
        if len(model_configs) != len(model_names):
            missing = set(model_names) - set(MODELS.keys())
            logger.warning(f"Unknown models ignored: {missing}")

    logger.info(f"Models to process: {[m.name for m in model_configs]}")

    # Select strategies
    if args.strategies == "all":
        strategies = ALL_STRATEGIES
    else:
        strategies = [s.strip() for s in args.strategies.split(",")]
        invalid = set(strategies) - set(ALL_STRATEGIES)
        if invalid:
            logger.warning(f"Unknown strategies ignored: {invalid}")
            strategies = [s for s in strategies if s in ALL_STRATEGIES]

    logger.info(f"Strategies to process: {strategies}")

    # Process all combinations
    total_start = time.time()
    results = []
    total_combinations = len(model_configs) * len(strategies)
    current = 0

    for config in model_configs:
        for strategy in strategies:
            current += 1
            logger.info(f"\nProcessing {current}/{total_combinations}...")
            try:
                result = process_model_strategy(
                    config,
                    entities,
                    output_dir,
                    device,
                    strategy,
                    args.batch_size,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {config.name} ({strategy}): {e}")
                import traceback

                traceback.print_exc()
                results.append(
                    {
                        "model": config.name,
                        "strategy": strategy,
                        "error": str(e),
                    }
                )
                clear_gpu_memory()

    # Save summary
    total_time = time.time() - total_start
    summary = {
        "total_time_seconds": total_time,
        "total_time_human": f"{total_time / 60:.1f} minutes",
        "entity_count": len(entities),
        "device": device,
        "models": [m.name for m in model_configs],
        "strategies": strategies,
        "results": results,
    }

    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time / 60:.1f} minutes")
    logger.info(f"Summary saved: {summary_path}")

    # Print summary
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    logger.info(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        logger.info(
            f"  {r['model']} ({r['strategy']}): {r['total_time']:.1f}s, "
            f"{r['num_vectors']} vectors"
        )

    if failed:
        logger.info(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            logger.info(f"  {r['model']} ({r['strategy']}): {r['error']}")


if __name__ == "__main__":
    main()
