# SPDX-License-Identifier: Apache-2.0
"""Diversity Agent Block for semantic clustering and diversity optimization.

This module implements the Diversity Agent from the multi-agent RAG evaluation framework,
which leverages clustering techniques to maximize topical coverage and semantic variability
in synthetic QA dataset generation.
"""

# Standard
from typing import Any, List, Optional, Union
import numpy as np

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Local
from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "DiversityAgentBlock",
    "rag_evaluation",
    "Diversity agent that uses clustering techniques to maximize topical coverage and semantic variability",
)
class DiversityAgentBlock(BaseBlock):
    """Diversity Agent for semantic clustering and diversity optimization.

    This block implements the Diversity Agent component of the multi-agent RAG evaluation
    framework. It uses sentence embeddings and clustering techniques to:
    1. Analyze semantic diversity of input documents
    2. Assign documents to diversity clusters
    3. Compute diversity scores for quality assessment
    4. Extract semantic features for downstream processing

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Input columns: ["document", "domain"]
    output_cols : List[str]
        Output columns: ["diversity_cluster", "diversity_score", "semantic_features"]
    num_clusters : int, optional
        Number of clusters for semantic diversity analysis, by default 10
    diversity_threshold : float, optional
        Minimum diversity score threshold, by default 0.7
    embedding_model : str, optional
        Sentence transformer model for embeddings, by default "sentence-transformers/all-MiniLM-L6-v2"
    clustering_method : str, optional
        Clustering algorithm to use, by default "kmeans"
    max_features : int, optional
        Maximum number of semantic features to extract, by default 50
    random_state : int, optional
        Random state for reproducibility, by default 42

    Examples
    --------
    >>> block = DiversityAgentBlock(
    ...     block_name="diversity_analysis",
    ...     input_cols=["document", "domain"],
    ...     output_cols=["diversity_cluster", "diversity_score", "semantic_features"],
    ...     num_clusters=8,
    ...     diversity_threshold=0.75
    ... )
    """

    # Core configuration
    num_clusters: int = Field(
        10, description="Number of clusters for semantic diversity analysis"
    )
    diversity_threshold: float = Field(
        0.7, description="Minimum diversity score threshold"
    )
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    clustering_method: str = Field(
        "kmeans", description="Clustering algorithm to use"
    )
    max_features: int = Field(
        50, description="Maximum number of semantic features to extract"
    )
    random_state: int = Field(
        42, description="Random state for reproducibility"
    )

    # Internal components (excluded from serialization)
    sentence_transformer: Optional[SentenceTransformer] = Field(None, exclude=True)
    scaler: Optional[StandardScaler] = Field(None, exclude=True)
    clusterer: Optional[Any] = Field(None, exclude=True)

    @field_validator("input_cols")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate input columns."""
        if v != ["document", "domain"]:
            raise BlockValidationError(
                f"DiversityAgentBlock requires input_cols=['document', 'domain'], got {v}"
            )
        return v

    @field_validator("output_cols")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate output columns."""
        expected = ["diversity_cluster", "diversity_score", "semantic_features"]
        if v != expected:
            raise BlockValidationError(
                f"DiversityAgentBlock requires output_cols={expected}, got {v}"
            )
        return v

    @field_validator("num_clusters")
    @classmethod
    def validate_num_clusters(cls, v):
        """Validate number of clusters."""
        if v < 2:
            raise BlockValidationError("num_clusters must be at least 2")
        return v

    @field_validator("diversity_threshold")
    @classmethod
    def validate_diversity_threshold(cls, v):
        """Validate diversity threshold."""
        if not 0.0 <= v <= 1.0:
            raise BlockValidationError("diversity_threshold must be between 0.0 and 1.0")
        return v

    @field_validator("clustering_method")
    @classmethod
    def validate_clustering_method(cls, v):
        """Validate clustering method."""
        supported_methods = ["kmeans"]
        if v not in supported_methods:
            raise BlockValidationError(
                f"clustering_method must be one of {supported_methods}, got {v}"
            )
        return v

    def _initialize_components(self):
        """Initialize the sentence transformer and clustering components."""
        if self.sentence_transformer is None:
            logger.info(f"Loading sentence transformer model: {self.embedding_model}")
            self.sentence_transformer = SentenceTransformer(self.embedding_model)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        
        if self.clusterer is None:
            if self.clustering_method == "kmeans":
                self.clusterer = KMeans(
                    n_clusters=self.num_clusters,
                    random_state=self.random_state,
                    n_init=10
                )

    def _compute_embeddings(self, documents: List[str]) -> np.ndarray:
        """Compute sentence embeddings for documents."""
        logger.info(f"Computing embeddings for {len(documents)} documents")
        embeddings = self.sentence_transformer.encode(
            documents,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings

    def _perform_clustering(self, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
        """Perform clustering on embeddings and return cluster labels and silhouette score."""
        logger.info(f"Performing {self.clustering_method} clustering with {self.num_clusters} clusters")
        
        # Standardize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Fit clustering model
        cluster_labels = self.clusterer.fit_predict(embeddings_scaled)
        
        # Compute silhouette score as diversity metric
        if len(set(cluster_labels)) > 1 and len(embeddings_scaled) > len(set(cluster_labels)):
            silhouette_avg = silhouette_score(embeddings_scaled, cluster_labels)
        else:
            # Not enough samples for silhouette score, use a simple diversity metric
            silhouette_avg = 0.5  # Default moderate diversity score
        
        logger.info(f"Clustering completed. Silhouette score: {silhouette_avg:.3f}")
        
        return cluster_labels, silhouette_avg

    def _compute_diversity_scores(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> List[float]:
        """Compute individual diversity scores for each document."""
        diversity_scores = []
        
        for i, (embedding, cluster_id) in enumerate(zip(embeddings, cluster_labels)):
            # Get embeddings of documents in the same cluster
            same_cluster_mask = cluster_labels == cluster_id
            same_cluster_embeddings = embeddings[same_cluster_mask]
            
            if len(same_cluster_embeddings) == 1:
                # Only document in cluster - maximum diversity
                diversity_score = 1.0
            else:
                # Compute average cosine distance to other documents in same cluster
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity([embedding], same_cluster_embeddings)[0]
                # Exclude self-similarity
                other_similarities = similarities[similarities < 0.999]
                if len(other_similarities) > 0:
                    avg_similarity = np.mean(other_similarities)
                    diversity_score = 1.0 - avg_similarity  # Convert similarity to diversity
                else:
                    diversity_score = 1.0
            
            diversity_scores.append(max(0.0, min(1.0, diversity_score)))  # Clamp to [0,1]
        
        return diversity_scores

    def _extract_semantic_features(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> List[List[float]]:
        """Extract semantic features for each document."""
        features_list = []
        
        for i, (embedding, cluster_id) in enumerate(zip(embeddings, cluster_labels)):
            # Use top-k dimensions of embedding as semantic features
            top_indices = np.argsort(np.abs(embedding))[-self.max_features:]
            semantic_features = embedding[top_indices].tolist()
            features_list.append(semantic_features)
        
        return features_list

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate diversity analysis for input documents.

        Parameters
        ----------
        samples : Dataset
            Input dataset with 'document' and 'domain' columns.

        Returns
        -------
        Dataset
            Dataset with added diversity analysis columns.
        """
        # Initialize components
        self._initialize_components()
        
        # Extract documents
        documents = samples["document"]
        domains = samples["domain"]
        
        logger.info(f"Processing {len(documents)} documents for diversity analysis")
        
        # Compute embeddings
        embeddings = self._compute_embeddings(documents)
        
        # Perform clustering
        cluster_labels, overall_diversity = self._perform_clustering(embeddings)
        
        # Compute individual diversity scores
        diversity_scores = self._compute_diversity_scores(embeddings, cluster_labels)
        
        # Extract semantic features
        semantic_features = self._extract_semantic_features(embeddings, cluster_labels)
        
        # Log statistics
        logger.info(f"Diversity analysis completed:")
        logger.info(f"  - Overall diversity (silhouette score): {overall_diversity:.3f}")
        logger.info(f"  - Average individual diversity: {np.mean(diversity_scores):.3f}")
        logger.info(f"  - Documents above threshold ({self.diversity_threshold}): "
                   f"{sum(1 for score in diversity_scores if score >= self.diversity_threshold)}")
        
        # Add results to dataset
        # Create new dataset with original data plus new columns
        new_data = {}
        for col in samples.column_names:
            new_data[col] = samples[col]
        
        new_data["diversity_cluster"] = cluster_labels.tolist()
        new_data["diversity_score"] = diversity_scores
        new_data["semantic_features"] = semantic_features
        
        return Dataset.from_dict(new_data)
