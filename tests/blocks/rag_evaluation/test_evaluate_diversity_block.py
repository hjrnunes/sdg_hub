#!/usr/bin/env python3
"""Tests for EvaluateDiversityBlock."""

import pytest
from datasets import Dataset
from unittest.mock import Mock, patch

from sdg_hub.core.blocks.rag_evaluation.evaluate_diversity_block import EvaluateDiversityBlock


class TestEvaluateDiversityBlock:
    """Test cases for EvaluateDiversityBlock."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            "document": [
                "Machine learning algorithms process data efficiently.",
                "Deep learning networks have multiple hidden layers.",
                "Natural language processing understands human text.",
                "Computer vision analyzes visual information.",
                "Reinforcement learning uses reward-based training."
            ],
            "domain": ["technology", "technology", "technology", "technology", "technology"],
            "document_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "diversity_cluster": [0, 1, 0, 1, 2],
            "diversity_score": [0.8, 0.7, 0.9, 0.6, 0.85],
            "semantic_features": [
                [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.1, 0.3, 0.2],
                [0.5, 0.4, 0.7], [0.2, 0.8, 0.1]
            ],
            "metadata": [{"source": f"test_{i}"} for i in range(5)]
        }
        self.dataset = Dataset.from_dict(self.test_data)

    def test_block_initialization(self):
        """Test block initialization with default parameters."""
        block = EvaluateDiversityBlock()
        
        assert block.diversity_threshold == 0.7
        assert block.cluster_balance_weight == 0.3
        assert block.semantic_spread_weight == 0.4
        assert block.coverage_weight == 0.3

    def test_block_initialization_with_params(self):
        """Test block initialization with custom parameters."""
        block = EvaluateDiversityBlock(
            diversity_threshold=0.8,
            cluster_balance_weight=0.4,
            semantic_spread_weight=0.3,
            coverage_weight=0.3
        )
        
        assert block.diversity_threshold == 0.8
        assert block.cluster_balance_weight == 0.4
        assert block.semantic_spread_weight == 0.3
        assert block.coverage_weight == 0.3

    def test_generate_success(self):
        """Test successful diversity evaluation."""
        block = EvaluateDiversityBlock()
        result = block.generate(self.dataset)
        
        # Check that result is a Dataset
        assert isinstance(result, Dataset)
        
        # Check that evaluation columns are added
        assert "diversity_evaluation_score" in result.column_names
        assert "cluster_balance_score" in result.column_names
        assert "semantic_spread_score" in result.column_names
        assert "coverage_score" in result.column_names
        assert "diversity_quality_assessment" in result.column_names
        
        # Check that original columns are preserved
        for col in self.dataset.column_names:
            assert col in result.column_names
        
        # Check data integrity
        assert len(result) == len(self.dataset)
        
        # Check score ranges
        for i in range(len(result)):
            assert 0 <= result["diversity_evaluation_score"][i] <= 1
            assert 0 <= result["cluster_balance_score"][i] <= 1
            assert 0 <= result["semantic_spread_score"][i] <= 1
            assert 0 <= result["coverage_score"][i] <= 1

    def test_generate_missing_required_columns(self):
        """Test generation with missing required columns."""
        invalid_data = {
            "document": ["Some text"],
            "domain": ["test"]
            # Missing diversity_cluster, diversity_score, semantic_features
        }
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = EvaluateDiversityBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block.generate(invalid_dataset)

    def test_calculate_cluster_balance(self):
        """Test cluster balance calculation."""
        block = EvaluateDiversityBlock()
        
        # Balanced clusters
        balanced_clusters = [0, 0, 1, 1, 2, 2]
        balance_score_balanced = block._calculate_cluster_balance(balanced_clusters)
        
        # Imbalanced clusters
        imbalanced_clusters = [0, 0, 0, 0, 1, 2]
        balance_score_imbalanced = block._calculate_cluster_balance(imbalanced_clusters)
        
        assert 0 <= balance_score_balanced <= 1
        assert 0 <= balance_score_imbalanced <= 1
        assert balance_score_balanced >= balance_score_imbalanced

    def test_calculate_semantic_spread(self):
        """Test semantic spread calculation."""
        block = EvaluateDiversityBlock()
        
        # High spread features (diverse)
        diverse_features = [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
            [0.1, 0.9, 0.1],
            [0.9, 0.1, 0.9]
        ]
        spread_score_high = block._calculate_semantic_spread(diverse_features)
        
        # Low spread features (similar)
        similar_features = [
            [0.5, 0.5, 0.5],
            [0.51, 0.49, 0.52],
            [0.49, 0.51, 0.48],
            [0.52, 0.48, 0.51]
        ]
        spread_score_low = block._calculate_semantic_spread(similar_features)
        
        assert 0 <= spread_score_high <= 1
        assert 0 <= spread_score_low <= 1
        assert spread_score_high >= spread_score_low

    def test_calculate_coverage_score(self):
        """Test coverage score calculation."""
        block = EvaluateDiversityBlock()
        
        # High diversity scores
        high_diversity = [0.9, 0.8, 0.85, 0.9, 0.75]
        coverage_high = block._calculate_coverage_score(high_diversity)
        
        # Low diversity scores
        low_diversity = [0.3, 0.2, 0.4, 0.1, 0.35]
        coverage_low = block._calculate_coverage_score(low_diversity)
        
        assert 0 <= coverage_high <= 1
        assert 0 <= coverage_low <= 1
        assert coverage_high >= coverage_low

    def test_assess_quality(self):
        """Test quality assessment."""
        block = EvaluateDiversityBlock()
        
        # High quality scores
        high_score = 0.85
        assessment_high = block._assess_quality(high_score)
        assert assessment_high in ["Excellent", "Good", "Fair", "Poor"]
        
        # Low quality scores
        low_score = 0.3
        assessment_low = block._assess_quality(low_score)
        assert assessment_low in ["Excellent", "Good", "Fair", "Poor"]
        
        # Threshold testing
        threshold_score = block.diversity_threshold
        assessment_threshold = block._assess_quality(threshold_score)
        assert assessment_threshold in ["Excellent", "Good", "Fair", "Poor"]

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        block = EvaluateDiversityBlock()
        
        # Should not raise any exception
        block._validate_input(self.dataset)

    def test_validate_input_missing_column(self):
        """Test input validation with missing required column."""
        invalid_data = {
            "document": ["test"],
            "domain": ["test"]
            # Missing diversity columns
        }
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = EvaluateDiversityBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block._validate_input(invalid_dataset)

    def test_weights_sum_to_one(self):
        """Test that evaluation weights sum to approximately 1.0."""
        block = EvaluateDiversityBlock()
        
        total_weight = (
            block.cluster_balance_weight + 
            block.semantic_spread_weight + 
            block.coverage_weight
        )
        
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point errors

    def test_evaluation_consistency(self):
        """Test that evaluation produces consistent results."""
        block = EvaluateDiversityBlock()
        
        # Run evaluation twice
        result1 = block.generate(self.dataset)
        result2 = block.generate(self.dataset)
        
        # Results should be identical
        for i in range(len(result1)):
            assert result1["diversity_evaluation_score"][i] == result2["diversity_evaluation_score"][i]
            assert result1["cluster_balance_score"][i] == result2["cluster_balance_score"][i]
            assert result1["semantic_spread_score"][i] == result2["semantic_spread_score"][i]
            assert result1["coverage_score"][i] == result2["coverage_score"][i]


if __name__ == "__main__":
    pytest.main([__file__])
