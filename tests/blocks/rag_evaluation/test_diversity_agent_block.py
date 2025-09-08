#!/usr/bin/env python3
"""Tests for DiversityAgentBlock."""

import pytest
import numpy as np
from datasets import Dataset
from unittest.mock import Mock, patch

from sdg_hub.core.blocks.rag_evaluation.diversity_agent_block import DiversityAgentBlock


class TestDiversityAgentBlock:
    """Test cases for DiversityAgentBlock."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            "document": [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "Deep learning uses neural networks with multiple layers to process data.",
                "Natural language processing helps computers understand human language.",
                "Computer vision enables machines to interpret and analyze visual information.",
                "Reinforcement learning trains agents through rewards and punishments."
            ],
            "domain": ["technology", "technology", "technology", "technology", "technology"],
            "document_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "metadata": [{"source": f"test_{i}"} for i in range(5)]
        }
        self.dataset = Dataset.from_dict(self.test_data)

    def test_block_initialization(self):
        """Test block initialization with default parameters."""
        block = DiversityAgentBlock()
        
        assert block.num_clusters == 3
        assert block.diversity_threshold == 0.5
        assert block.embedding_model == "all-MiniLM-L6-v2"
        assert block.clustering_method == "kmeans"

    def test_block_initialization_with_params(self):
        """Test block initialization with custom parameters."""
        block = DiversityAgentBlock(
            num_clusters=5,
            diversity_threshold=0.7,
            embedding_model="all-mpnet-base-v2",
            clustering_method="kmeans"
        )
        
        assert block.num_clusters == 5
        assert block.diversity_threshold == 0.7
        assert block.embedding_model == "all-mpnet-base-v2"
        assert block.clustering_method == "kmeans"

    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.SentenceTransformer')
    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.KMeans')
    def test_generate_success(self, mock_kmeans, mock_sentence_transformer):
        """Test successful generation with mocked dependencies."""
        # Mock SentenceTransformer
        mock_transformer = Mock()
        mock_transformer.encode.return_value = np.random.rand(5, 384)  # 5 docs, 384 dims
        mock_sentence_transformer.return_value = mock_transformer
        
        # Mock KMeans
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 1, 0, 1, 2])
        mock_clusterer.labels_ = np.array([0, 1, 0, 1, 2])
        mock_kmeans.return_value = mock_clusterer
        
        block = DiversityAgentBlock(num_clusters=3)
        result = block.generate(self.dataset)
        
        # Check that result is a Dataset
        assert isinstance(result, Dataset)
        
        # Check that new columns are added
        assert "diversity_cluster" in result.column_names
        assert "diversity_score" in result.column_names
        assert "semantic_features" in result.column_names
        
        # Check that original columns are preserved
        for col in self.dataset.column_names:
            assert col in result.column_names
        
        # Check data integrity
        assert len(result) == len(self.dataset)

    def test_generate_insufficient_data(self):
        """Test generation with insufficient data."""
        small_data = {
            "document": ["Single document"],
            "domain": ["test"],
            "document_id": ["doc1"],
            "metadata": [{"source": "test"}]
        }
        small_dataset = Dataset.from_dict(small_data)
        
        block = DiversityAgentBlock(num_clusters=3)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            block.generate(small_dataset)

    def test_generate_missing_required_columns(self):
        """Test generation with missing required columns."""
        invalid_data = {
            "text": ["Some text"],  # Wrong column name
            "domain": ["test"]
        }
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = DiversityAgentBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block.generate(invalid_dataset)

    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.SentenceTransformer')
    def test_compute_embeddings(self, mock_sentence_transformer):
        """Test embedding computation."""
        mock_transformer = Mock()
        mock_transformer.encode.return_value = np.random.rand(5, 384)
        mock_sentence_transformer.return_value = mock_transformer
        
        block = DiversityAgentBlock()
        block._initialize_components()
        
        embeddings = block._compute_embeddings(self.test_data["document"])
        
        assert embeddings.shape == (5, 384)
        mock_transformer.encode.assert_called_once()

    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.KMeans')
    def test_perform_clustering(self, mock_kmeans):
        """Test clustering performance."""
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 1, 0, 1, 2])
        mock_clusterer.labels_ = np.array([0, 1, 0, 1, 2])
        mock_kmeans.return_value = mock_clusterer
        
        block = DiversityAgentBlock(num_clusters=3)
        block._initialize_components()
        
        embeddings = np.random.rand(5, 384)
        cluster_labels, diversity_score = block._perform_clustering(embeddings)
        
        assert len(cluster_labels) == 5
        assert isinstance(diversity_score, float)
        assert 0 <= diversity_score <= 1

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        block = DiversityAgentBlock()
        
        # Should not raise any exception
        block._validate_input(self.dataset)

    def test_validate_input_missing_column(self):
        """Test input validation with missing required column."""
        invalid_data = {"domain": ["test"]}
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = DiversityAgentBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block._validate_input(invalid_dataset)

    def test_validate_input_insufficient_data(self):
        """Test input validation with insufficient data."""
        small_data = {
            "document": ["Single doc"],
            "domain": ["test"],
            "document_id": ["doc1"],
            "metadata": [{"source": "test"}]
        }
        small_dataset = Dataset.from_dict(small_data)
        
        block = DiversityAgentBlock(num_clusters=3)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            block._validate_input(small_dataset)


if __name__ == "__main__":
    pytest.main([__file__])
