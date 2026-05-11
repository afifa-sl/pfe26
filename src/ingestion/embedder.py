"""Génération d'embeddings locaux via sentence-transformers."""
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        logger.info("Chargement du modèle d'embedding: %s...", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("  Dimension: %d | Device: %s", self.dimension, device)

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed une liste de textes. Retourne un array (N, dim)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Normalisation L2 pour cosine similarity
            convert_to_numpy=True,
        )

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
