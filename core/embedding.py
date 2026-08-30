"""
Dense Vector Embedding Module for BugTrace AI.

Uses sentence-transformers (all-MiniLM-L6-v2, 384-dim) to generate dense vector
embeddings for issue contexts and search queries with hardware acceleration.
"""

import logging
from typing import List, Optional, Union

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("EmbeddingGenerator")


class EmbeddingGenerator:
    """
    Manages loading the sentence-transformers model and computing embeddings.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the embedding generator.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run inference on ('cuda', 'mps', 'cpu'). Auto-detected if None.
            normalize_embeddings: Whether to L2-normalize vectors for cosine similarity.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.device = device or self._detect_device()
        self._model: Optional[SentenceTransformer] = None
        self._dimension = self.DEFAULT_DIMENSION

        logger.info(
            f"Initializing EmbeddingGenerator with model='{self.model_name}' on device='{self.device}'"
        )

    @staticmethod
    def _detect_device() -> str:
        """
        Detect optimal available hardware accelerator.
        """
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the SentenceTransformer model on first access.
        """
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' into memory...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Retrieve model output dimension dynamically
            test_emb = self._model.encode("test", convert_to_numpy=True)
            self._dimension = test_emb.shape[-1]
            logger.info(
                f"Model loaded successfully. Output vector dimension: {self._dimension}"
            )
        return self._model

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.
        """
        return self._dimension

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate dense embeddings for a batch of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.
            show_progress: Whether to display progress bar.

        Returns:
            List of 384-dimensional float lists.
        """
        if not texts:
            return []

        logger.info(
            f"Generating embeddings for {len(texts)} texts (batch_size={batch_size}, device={self.device})..."
        )
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single search query.

        Args:
            text: Query string.

        Returns:
            384-dimensional float vector.
        """
        if not text or not text.strip():
            # Return zero vector of appropriate dimension for empty queries
            return [0.0] * self.dimension

        embedding = self.model.encode(
            text.strip(),
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embedding.tolist()
