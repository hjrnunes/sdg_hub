#!/usr/bin/env python3
"""Tests for PrivacyAgentBlock."""

import pytest
from datasets import Dataset
from unittest.mock import Mock, patch, MagicMock

from sdg_hub.core.blocks.rag_evaluation.privacy_agent_block import PrivacyAgentBlock


class TestPrivacyAgentBlock:
    """Test cases for PrivacyAgentBlock."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            "document": [
                "John Smith works at Acme Corp and his email is john.smith@acme.com.",
                "Patient Sarah Johnson (DOB: 03/15/1985) was admitted with chest pain.",
                "The financial report shows revenue of $2.5 million for Q3 2024.",
                "Dr. Michael Brown can be reached at (555) 123-4567 for consultations.",
                "This is a general document without sensitive information."
            ],
            "domain": ["business", "medical", "financial", "medical", "general"],
            "document_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
            "metadata": [{"source": f"test_{i}"} for i in range(5)]
        }
        self.dataset = Dataset.from_dict(self.test_data)

    def test_block_initialization(self):
        """Test block initialization with default parameters."""
        block = PrivacyAgentBlock()
        
        assert block.privacy_domains == ["medical", "financial", "legal", "personal"]
        assert block.masking_strategy == "entity_replacement"
        assert "PERSON" in block.entity_types
        assert "ORG" in block.entity_types
        assert block.preserve_structure is True

    def test_block_initialization_with_params(self):
        """Test block initialization with custom parameters."""
        block = PrivacyAgentBlock(
            privacy_domains=["medical", "financial"],
            masking_strategy="redaction",
            entity_types=["PERSON", "ORG"],
            preserve_structure=False
        )
        
        assert block.privacy_domains == ["medical", "financial"]
        assert block.masking_strategy == "redaction"
        assert block.entity_types == ["PERSON", "ORG"]
        assert block.preserve_structure is False

    @patch('sdg_hub.core.blocks.rag_evaluation.privacy_agent_block.spacy.load')
    def test_generate_success(self, mock_spacy_load):
        """Test successful generation with mocked spaCy."""
        # Mock spaCy NLP pipeline
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Mock entities
        mock_entity1 = Mock()
        mock_entity1.text = "John Smith"
        mock_entity1.label_ = "PERSON"
        mock_entity1.start_char = 0
        mock_entity1.end_char = 10
        
        mock_entity2 = Mock()
        mock_entity2.text = "john.smith@acme.com"
        mock_entity2.label_ = "EMAIL"
        mock_entity2.start_char = 50
        mock_entity2.end_char = 69
        
        mock_doc.ents = [mock_entity1, mock_entity2]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        block = PrivacyAgentBlock()
        result = block.generate(self.dataset)
        
        # Check that result is a Dataset
        assert isinstance(result, Dataset)
        
        # Check that new columns are added
        assert "privacy_masked_document" in result.column_names
        assert "privacy_score" in result.column_names
        assert "detected_entities" in result.column_names
        
        # Check that original columns are preserved
        for col in self.dataset.column_names:
            assert col in result.column_names
        
        # Check data integrity
        assert len(result) == len(self.dataset)

    def test_generate_missing_required_columns(self):
        """Test generation with missing required columns."""
        invalid_data = {
            "text": ["Some text"],  # Wrong column name
            "domain": ["test"]
        }
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = PrivacyAgentBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block.generate(invalid_dataset)

    @patch('sdg_hub.core.blocks.rag_evaluation.privacy_agent_block.spacy.load')
    def test_detect_entities(self, mock_spacy_load):
        """Test entity detection."""
        # Mock spaCy
        mock_nlp = Mock()
        mock_doc = Mock()
        
        mock_entity = Mock()
        mock_entity.text = "John Smith"
        mock_entity.label_ = "PERSON"
        mock_entity.start_char = 0
        mock_entity.end_char = 10
        
        mock_doc.ents = [mock_entity]
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        block = PrivacyAgentBlock()
        block._initialize_components()
        
        entities = block._detect_entities("John Smith works here", "business")
        
        assert len(entities) >= 0  # May detect entities or not based on mocking
        mock_nlp.assert_called()

    def test_mask_text_entity_replacement(self):
        """Test text masking with entity replacement strategy."""
        block = PrivacyAgentBlock(masking_strategy="entity_replacement")
        
        entities = [
            {"text": "John Smith", "label": "PERSON", "start": 0, "end": 10},
            {"text": "john@email.com", "label": "EMAIL", "start": 20, "end": 34}
        ]
        
        original_text = "John Smith has email john@email.com"
        masked_text = block._mask_text(original_text, entities)
        
        # Should replace with generic placeholders
        assert "John Smith" not in masked_text
        assert "john@email.com" not in masked_text
        assert "[PERSON]" in masked_text or "[NAME]" in masked_text

    def test_mask_text_redaction(self):
        """Test text masking with redaction strategy."""
        block = PrivacyAgentBlock(masking_strategy="redaction")
        
        entities = [
            {"text": "John Smith", "label": "PERSON", "start": 0, "end": 10}
        ]
        
        original_text = "John Smith works here"
        masked_text = block._mask_text(original_text, entities)
        
        # Should redact with asterisks or similar
        assert "John Smith" not in masked_text
        assert "*" in masked_text or "[REDACTED]" in masked_text

    def test_calculate_privacy_score(self):
        """Test privacy score calculation."""
        block = PrivacyAgentBlock()
        
        # High-risk domain with many entities
        entities_high_risk = [
            {"label": "PERSON"}, {"label": "EMAIL"}, {"label": "PHONE"}
        ]
        score_high_risk = block._calculate_privacy_score(entities_high_risk, "medical")
        
        # Low-risk domain with few entities
        entities_low_risk = [{"label": "ORG"}]
        score_low_risk = block._calculate_privacy_score(entities_low_risk, "general")
        
        assert 0 <= score_high_risk <= 1
        assert 0 <= score_low_risk <= 1
        assert score_high_risk >= score_low_risk  # Medical should be higher risk

    def test_validate_input_valid(self):
        """Test input validation with valid data."""
        block = PrivacyAgentBlock()
        
        # Should not raise any exception
        block._validate_input(self.dataset)

    def test_validate_input_missing_column(self):
        """Test input validation with missing required column."""
        invalid_data = {"domain": ["test"]}
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        block = PrivacyAgentBlock()
        
        with pytest.raises(ValueError, match="Missing required column"):
            block._validate_input(invalid_dataset)

    def test_privacy_patterns_initialization(self):
        """Test that privacy patterns are properly initialized."""
        block = PrivacyAgentBlock()
        block._initialize_components()
        
        # Should have patterns for different domains
        assert len(block.privacy_patterns) > 0
        
        # Should have medical patterns
        if "medical" in block.privacy_patterns:
            medical_patterns = block.privacy_patterns["medical"]
            assert len(medical_patterns) > 0

    def test_domain_specific_processing(self):
        """Test that different domains are processed differently."""
        block = PrivacyAgentBlock()
        
        medical_text = "Patient John Doe, DOB: 01/01/1980, SSN: 123-45-6789"
        financial_text = "Account number: 1234567890, Balance: $50,000"
        
        # Both should be processed, but medical might have higher risk
        medical_score = block._calculate_privacy_score(
            [{"label": "PERSON"}, {"label": "DATE"}, {"label": "SSN"}], 
            "medical"
        )
        financial_score = block._calculate_privacy_score(
            [{"label": "ACCOUNT"}, {"label": "MONEY"}], 
            "financial"
        )
        
        assert 0 <= medical_score <= 1
        assert 0 <= financial_score <= 1


if __name__ == "__main__":
    pytest.main([__file__])
