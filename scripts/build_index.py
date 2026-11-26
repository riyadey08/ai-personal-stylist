"""Build the StandardScaler + PCA + FAISS artifacts from the bundled CSV.

Usage: python scripts/build_index.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.retrieval import StyleIndex


def main() -> None:
    config.ensure_dirs()
    style_index = StyleIndex.build(config.TRAINING_DATA_PATH, config.MODELS_DIR)
    print(f"Indexed {style_index.index.ntotal} vectors in {style_index.index.d} dimensions.")
    print(f"Artifacts written to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
