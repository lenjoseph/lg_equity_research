"""
Runs once after installation to download embedding model from huggingface.
Cached post-installation.
Usage: python scripts/warmup_embeddings.py
"""

import sys
from pathlib import Path
from huggingface_hub import logging as hf_logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.shared.embedding_models import get_embeddings

hf_logging.set_verbosity_info()

if __name__ == "__main__":
    print("Downloading embedding model...")
    embeddings = get_embeddings()
    test_embedding = embeddings.embed_query("test")
    print(f"model loaded: embedding dimensions: {len(test_embedding)}")
