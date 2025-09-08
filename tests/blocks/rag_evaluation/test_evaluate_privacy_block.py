#!/usr/bin/env python3
"""Tests for EvaluatePrivacyBlock."""

import pytest
from datasets import Dataset
from unittest.mock import Mock, patch

from sdg_hub.core.blocks.rag_evaluation.evaluate_privacy_block import EvaluatePrivacyBlock


class TestEvaluatePrivacyBlock:
    """Test cases for EvaluatePrivacyBlock."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            "document": [
                "John Smith works at Acme Corp and his email is john.smith@acme.com.",
                "Patient [PATIENT_NAME] was admitted with chest pain on [DATE].",
                "The financial report shows revenue of $2.5 million for Q3 2024.",
                "Dr. [DOCTOR_NAME] can be reached at [PHONE] for consultations.",
                "This is a general document without sensitive information."
            ],
            "privacy_masked_document": [
                "[PERSON] works at [ORG] and his email is [EMAIL].",
                "Patient [PATIENT_NAME] was admitted with chest pain on [DATE].",
                "The financial report shows revenue of [MONEY] for [DATE].",
                "Dr. [DOCTOR_NAME] can be reached at [PHONE] for consultations.",
                "This is a general document without sensitive information."
            ],
            "domain": ["business", "medical", "financial", "medical", "general"],
            "privacy_score": [0.8, 0.9, 0.7, 0.85, 0.1],
            "detected_entities": [
                [{"text": "John Smith", "label": "PERSON"}, {"text": "john.smith@acme.com", "label": "EMAIL"}],
                [{"text": "Patient Name", "label": "PERSON"}, {"text": "03/15/1985", "label": "DATE"}],
                [{"text": "$2.5 million", "label": "MONEY"}, {"text": "Q3 2024", "label": "DATE"}],
                [{"text": "Dr. Brown", "label": "PERSON"}, {"text": "(555) 123-4567", "label": "PHONE"}],
                []
            ],
            "document_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "metadata": [{"source": f"test_{i}"} for i in range(5)]
        }
        self.dataset = Dataset.from_dict(self.test_data)

    def test_block_initialization(self):
        """Test block initialization with default parameters."""
        block = EvaluatePrivacyBlock()
        
        assert block.privacy_threshold == 0.8
        assert block.masking_effectiveness_weight == 0.4
        assert block.entity_coverage_weight == 0.3
        assert block.domain_compliance_weight == 0.3

    def test_block_initialization_with_params(self):
        """Test block initialization with custom parameters."""
        block = EvaluatePrivacyBlock(
            privacy_threshold=0.9,
            masking_effectiveness_weight=0.5,
            entity_coverage_weight=0.25,
            domain_compliance_weight=0.25
        )
        
        assert block.privacy_threshold == 0.9
        assert block.masking_effectiveness_weight == 0.5
        assert block.entity_coverage_weight == 0.25
        assert block.domain_compliance_weight == 0.25

    def test_generate_success(self):
        """Test successful privacy evaluation."""
        block = EvaluatePrivacyBlock()
        result = block.generate(self.dataset)
        
        # Check that result is a Dataset
        assert isinstance(result, Dataset)
        
        # Check that evaluation columns are added
        assert "privacy_evaluation_score" in result.column_names
        assert "masking_effectiveness_score" in result.column_names
        assert "entity_coverage_score" in result.column_names
        assert "domain_compliance_score" in result.column_names
        assert "privacy_quality_assessment" in result.column_names
        
        # Check that original columns are preserved
        for col in self.dataset.column_names:
            assert col in result.column_names
        
        # Check data integrity
        assert len(result) == len(self.dataset)
        
        # Check score ranges
        for i in range(len(result)):
            assert 0 <= result["privacy_evaluation_score"][i] <= 1
            assert 0 <= result["masking_effectiveness_score"][i] <= 1
            assert 0 <= result["entity_coverage_score"][i] <= 1
            assert 0 <= result["domain_compliance_score"][i] <= 1

    def test_generate_missing_required_columns(self):
        """Test generation with missing required columns."""
        invalid_data = {
            "document": ["Some text"],
            "domain": ["test"]
            # Missing privacy columns
        }
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = EvaluatePrivacyBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block.generate(invalid_dataset)

    def test_calculate_masking_effectiveness(self):
        """Test masking effectiveness calculation."""
        block = EvaluatePrivacyBlock()
        
        # Well-masked text
        original = "John Smith works at Acme Corp"
        masked = "[PERSON] works at [ORG]"
        entities = [
            {"text": "John Smith", "label": "PERSON"},
            {"text": "Acme Corp", "label": "ORG"}
        ]
        effectiveness_high = block._calculate_masking_effectiveness(original, masked, entities)
        
        # Poorly masked text
        poorly_masked = "John works at [ORG]"  # Name partially visible
        effectiveness_low = block._calculate_masking_effectiveness(original, poorly_masked, entities)
        
        assert 0 <= effectiveness_high <= 1
        assert 0 <= effectiveness_low <= 1
        assert effectiveness_high >= effectiveness_low

    def test_calculate_entity_coverage(self):
        """Test entity coverage calculation."""
        block = EvaluatePrivacyBlock()
        
        # High-risk entities detected
        high_risk_entities = [
            {"label": "PERSON"}, {"label": "SSN"}, {"label": "EMAIL"}, {"label": "PHONE"}
        ]
        coverage_high = block._calculate_entity_coverage(high_risk_entities, "medical")
        
        # Low-risk entities detected
        low_risk_entities = [
            {"label": "ORG"}
        ]
        coverage_low = block._calculate_entity_coverage(low_risk_entities, "general")
        
        assert 0 <= coverage_high <= 1
        assert 0 <= coverage_low <= 1

    def test_calculate_domain_compliance(self):
        """Test domain compliance calculation."""
        block = EvaluatePrivacyBlock()
        
        # Medical domain with appropriate privacy score
        medical_compliance = block._calculate_domain_compliance(0.9, "medical")
        
        # General domain with lower privacy score
        general_compliance = block._calculate_domain_compliance(0.3, "general")
        
        # Financial domain with high privacy score
        financial_compliance = block._calculate_domain_compliance(0.85, "financial")
        
        assert 0 <= medical_compliance <= 1
        assert 0 <= general_compliance <= 1
        assert 0 <= financial_compliance <= 1

    def test_assess_privacy_quality(self):
        """Test privacy quality assessment."""
        block = EvaluatePrivacyBlock()
        
        # High privacy score
        high_score = 0.9
        assessment_high = block._assess_privacy_quality(high_score)
        assert assessment_high in ["Excellent", "Good", "Fair", "Poor"]
        
        # Low privacy score
        low_score = 0.3
        assessment_low = block._assess_privacy_quality(low_score)
        assert assessment_low in ["Excellent", "Good", "Fair", "Poor"]
        
        # Threshold testing
        threshold_score = block.privacy_threshold
        assessment_threshold = block._assess_privacy_quality(threshold_score)
        assert assessment_threshold in ["Excellent", "Good", "Fair", "Poor"]

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        block = EvaluatePrivacyBlock()
        
        # Should not raise any exception
        block._validate_input(self.dataset)

    def test_validate_input_missing_column(self):
        """Test input validation with missing required column."""
        invalid_data = {
            "document": ["test"],
            "domain": ["test"]
            # Missing privacy columns
        }
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = EvaluatePrivacyBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block._validate_input(invalid_dataset)

    def test_weights_sum_to_one(self):
        """Test that evaluation weights sum to approximately 1.0."""
        block = EvaluatePrivacyBlock()
        
        total_weight = (
            block.masking_effectiveness_weight + 
            block.entity_coverage_weight + 
            block.domain_compliance_weight
        )
        
        assert abs(total_weight - 1.0) < 0.01  # Allow small floating point errors

    def test_sensitive_entity_detection(self):
        """Test detection of sensitive entity types."""
        block = EvaluatePrivacyBlock()
        
        # High-sensitivity entities
        sensitive_entities = [
            {"label": "SSN"}, {"label": "CREDIT_CARD"}, {"label": "PHONE"}
        ]
        
        # Low-sensitivity entities
        general_entities = [
            {"label": "ORG"}, {"label": "GPE"}
        ]
        
        sensitive_score = block._calculate_entity_coverage(sensitive_entities, "medical")
        general_score = block._calculate_entity_coverage(general_entities, "general")
        
        # Both should be valid scores
        assert 0 <= sensitive_score <= 1
        assert 0 <= general_score <= 1

    def test_domain_specific_requirements(self):
        """Test that different domains have different privacy requirements."""
        block = EvaluatePrivacyBlock()
        
        same_score = 0.7
        
        # Different domains should potentially have different compliance scores
        medical_compliance = block._calculate_domain_compliance(same_score, "medical")
        financial_compliance = block._calculate_domain_compliance(same_score, "financial")
        general_compliance = block._calculate_domain_compliance(same_score, "general")
        
        assert 0 <= medical_compliance <= 1
        assert 0 <= financial_compliance <= 1
        assert 0 <= general_compliance <= 1

    def test_evaluation_consistency(self):
        """Test that evaluation produces consistent results."""
        block = EvaluatePrivacyBlock()
        
        # Run evaluation twice
        result1 = block.generate(self.dataset)
        result2 = block.generate(self.dataset)
        
        # Results should be identical
        for i in range(len(result1)):
            assert result1["privacy_evaluation_score"][i] == result2["privacy_evaluation_score"][i]
            assert result1["masking_effectiveness_score"][i] == result2["masking_effectiveness_score"][i]
            assert result1["entity_coverage_score"][i] == result2["entity_coverage_score"][i]
            assert result1["domain_compliance_score"][i] == result2["domain_compliance_score"][i]


if __name__ == "__main__":
    pytest.main([__file__])
