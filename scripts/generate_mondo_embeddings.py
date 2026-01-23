#!/usr/bin/env python
"""Generate MONDO ontology embeddings.

Downloads MONDO OBO file if not present, parses to BaseEntity objects,
generates embeddings with biomedical models, and builds FAISS index.

Usage:
    # Auto-download MONDO and generate embeddings
    uv run python scripts/generate_mondo_embeddings.py --download

    # Use existing MONDO OBO file
    uv run python scripts/generate_mondo_embeddings.py --mondo-path data/raw/mondo/mondo.obo

    # Specify output directory
    uv run python scripts/generate_mondo_embeddings.py --download --output-dir data/mondo

    # Use a different embedding model
    uv run python scripts/generate_mondo_embeddings.py --download --model sapbert

Available models:
    - sapbert: SapBERT (default) - best for biomedical concepts
    - pubmedbert: PubMedBERT - good alternative
    - minilm: MiniLM - fastest, lower quality
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from biovector_eval.base.entity import BaseEntity

# MONDO release URL (latest release)
MONDO_RELEASE_URL = "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo"

# Embedding models
EMBEDDING_MODELS = {
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


def download_mondo(output_path: Path) -> None:
    """Download MONDO OBO file from GitHub releases.

    Args:
        output_path: Path to save the OBO file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading MONDO from {MONDO_RELEASE_URL}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(MONDO_RELEASE_URL, output_path)
        logger.info(f"Downloaded MONDO to {output_path}")
    except Exception as e:
        logger.error(f"Failed to download MONDO: {e}")
        raise


def parse_obo_file(obo_path: Path) -> list[BaseEntity]:
    """Parse MONDO OBO file into BaseEntity objects.

    OBO format consists of stanzas like:
    [Term]
    id: MONDO:0005015
    name: diabetes mellitus
    def: "A metabolic disorder..."
    synonym: "diabetes" EXACT []

    Args:
        obo_path: Path to MONDO OBO file.

    Returns:
        List of BaseEntity objects.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing MONDO OBO file: {obo_path}")

    entities: list[BaseEntity] = []
    current_term: dict = {}

    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # New term stanza
            if line == "[Term]":
                if current_term.get("id"):
                    entity = _term_to_entity(current_term)
                    if entity:
                        entities.append(entity)
                current_term = {}
                continue

            # Skip non-term stanzas
            if line.startswith("[") and line.endswith("]"):
                if current_term.get("id"):
                    entity = _term_to_entity(current_term)
                    if entity:
                        entities.append(entity)
                current_term = {}
                continue

            # Parse tag-value pairs
            if ":" not in line:
                continue

            tag, _, value = line.partition(":")
            value = value.strip()

            if tag == "id":
                current_term["id"] = value
            elif tag == "name":
                current_term["name"] = value
            elif tag == "def":
                # Extract definition text from quotes
                if value.startswith('"'):
                    end_quote = value.find('"', 1)
                    if end_quote > 0:
                        current_term["definition"] = value[1:end_quote]
            elif tag == "synonym":
                # Parse synonym: "text" SCOPE []
                if value.startswith('"'):
                    end_quote = value.find('"', 1)
                    if end_quote > 0:
                        synonym = value[1:end_quote]
                        if "synonyms" not in current_term:
                            current_term["synonyms"] = []
                        current_term["synonyms"].append(synonym)
            elif tag == "is_obsolete" and value == "true":
                current_term["obsolete"] = True
            elif tag == "namespace":
                current_term["namespace"] = value

        # Don't forget last term
        if current_term.get("id"):
            entity = _term_to_entity(current_term)
            if entity:
                entities.append(entity)

    logger.info(f"Parsed {len(entities)} MONDO terms")
    return entities


def _term_to_entity(term: dict) -> BaseEntity | None:
    """Convert parsed OBO term to BaseEntity.

    Args:
        term: Parsed term dict.

    Returns:
        BaseEntity or None if term should be skipped.
    """
    # Skip obsolete terms
    if term.get("obsolete"):
        return None

    # Skip terms without names
    if not term.get("name"):
        return None

    # Skip non-MONDO IDs (e.g., HP:, OMIM:)
    term_id = term.get("id", "")
    if not term_id.startswith("MONDO:"):
        return None

    return BaseEntity(
        id=term_id,
        name=term["name"],
        synonyms=term.get("synonyms", []),
        metadata={
            "ontology": "MONDO",
            "definition": term.get("definition", ""),
            "namespace": term.get("namespace", ""),
        },
    )


def generate_embeddings(
    entities: list[BaseEntity],
    model_name: str,
    batch_size: int = 32,
) -> "numpy.ndarray":
    """Generate embeddings for MONDO entities.

    Args:
        entities: List of BaseEntity objects.
        model_name: Name of embedding model to use.
        batch_size: Batch size for embedding generation.

    Returns:
        NumPy array of embeddings (n_entities x embedding_dim).
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    logger = logging.getLogger(__name__)

    model_id = EMBEDDING_MODELS.get(model_name, model_name)
    logger.info(f"Loading embedding model: {model_id}")

    model = SentenceTransformer(model_id)

    # Extract texts for embedding
    texts = [e.name for e in entities]
    logger.info(f"Generating embeddings for {len(texts)} terms")

    # Generate embeddings in batches with progress bar
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, normalize_embeddings=True)
        all_embeddings.append(embeddings)

    embeddings_array = np.vstack(all_embeddings).astype(np.float32)
    logger.info(f"Generated embeddings with shape {embeddings_array.shape}")

    return embeddings_array


def build_faiss_index(
    embeddings: "numpy.ndarray",
    index_type: str = "flat",
) -> "faiss.Index":
    """Build FAISS index from embeddings.

    Args:
        embeddings: NumPy array of embeddings.
        index_type: Type of index ("flat" or "hnsw").

    Returns:
        FAISS index.
    """
    import faiss

    logger = logging.getLogger(__name__)
    dim = embeddings.shape[1]

    if index_type == "hnsw":
        # HNSW index for faster search
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 16
    else:
        # Flat index for exact search (inner product for normalized vectors)
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)
    logger.info(f"Built FAISS {index_type} index with {index.ntotal} vectors")

    return index


def save_outputs(
    entities: list[BaseEntity],
    embeddings: "numpy.ndarray",
    index: "faiss.Index",
    output_dir: Path,
    model_name: str,
) -> dict[str, Path]:
    """Save entities, embeddings, and index to files.

    Args:
        entities: List of BaseEntity objects.
        embeddings: NumPy array of embeddings.
        index: FAISS index.
        output_dir: Output directory.
        model_name: Name of embedding model used.

    Returns:
        Dict mapping output type to file path.
    """
    import faiss
    import numpy as np

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Save entities JSON
    entities_path = output_dir / "mondo_entities.json"
    entities_data = [e.to_dict() for e in entities]
    with open(entities_path, "w", encoding="utf-8") as f:
        json.dump(entities_data, f, indent=2)
    outputs["entities"] = entities_path
    logger.info(f"Saved {len(entities)} entities to {entities_path}")

    # Save embeddings
    embeddings_path = output_dir / f"mondo_embeddings_{model_name}.npy"
    np.save(embeddings_path, embeddings)
    outputs["embeddings"] = embeddings_path
    logger.info(f"Saved embeddings to {embeddings_path}")

    # Save FAISS index
    index_path = output_dir / f"mondo_{model_name}.faiss"
    faiss.write_index(index, str(index_path))
    outputs["index"] = index_path
    logger.info(f"Saved FAISS index to {index_path}")

    # Save metadata
    metadata_path = output_dir / "mondo_metadata.json"
    metadata = {
        "model_name": model_name,
        "model_id": EMBEDDING_MODELS.get(model_name, model_name),
        "num_entities": len(entities),
        "embedding_dim": embeddings.shape[1],
        "index_type": "flat",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    outputs["metadata"] = metadata_path

    return outputs


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate MONDO ontology embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download MONDO and generate embeddings with SapBERT
    uv run python scripts/generate_mondo_embeddings.py --download

    # Use existing MONDO file
    uv run python scripts/generate_mondo_embeddings.py --mondo-path data/raw/mondo/mondo.obo

    # Use a faster model
    uv run python scripts/generate_mondo_embeddings.py --download --model minilm
        """,
    )
    parser.add_argument(
        "--mondo-path",
        type=Path,
        help="Path to existing MONDO OBO file",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MONDO OBO file from GitHub releases",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=project_root / "data" / "mondo",
        help="Output directory for embeddings and index (default: data/mondo)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(EMBEDDING_MODELS.keys()),
        default="sapbert",
        help="Embedding model to use (default: sapbert)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Determine MONDO file path
    if args.download:
        mondo_path = args.output_dir / "raw" / "mondo.obo"
        if not mondo_path.exists():
            download_mondo(mondo_path)
        else:
            logger.info(f"Using existing MONDO file: {mondo_path}")
    elif args.mondo_path:
        mondo_path = args.mondo_path
        if not mondo_path.exists():
            logger.error(f"MONDO file not found: {mondo_path}")
            return 1
    else:
        logger.error("Either --mondo-path or --download must be specified")
        return 1

    # Parse MONDO
    entities = parse_obo_file(mondo_path)
    if not entities:
        logger.error("No entities parsed from MONDO file")
        return 1

    # Generate embeddings
    try:
        embeddings = generate_embeddings(
            entities,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Install with: uv add sentence-transformers faiss-cpu tqdm")
        return 1

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save outputs
    outputs = save_outputs(
        entities,
        embeddings,
        index,
        args.output_dir,
        args.model,
    )

    print("\nSuccess! MONDO embeddings generated.")
    print(f"  Entities: {outputs['entities']}")
    print(f"  Embeddings: {outputs['embeddings']}")
    print(f"  FAISS index: {outputs['index']}")
    print(f"\nUse these paths with MONDOAdapter:")
    print(f"  mondo_index_path='{outputs['index']}'")
    print(f"  mondo_entities_path='{outputs['entities']}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
