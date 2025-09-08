#!/usr/bin/env python3
"""Integration tests for RAG evaluation blocks."""

import pytest
from datasets import Dataset
from unittest.mock import Mock, patch
import numpy as np

from sdg_hub.core.blocks.rag_evaluation.diversity_agent_block import DiversityAgentBlock
from sdg_hub.core.blocks.rag_evaluation.privacy_agent_block import PrivacyAgentBlock
from sdg_hub.core.blocks.rag_evaluation.evaluate_diversity_block import EvaluateDiversityBlock
from sdg_hub.core.blocks.rag_evaluation.evaluate_privacy_block import EvaluatePrivacyBlock


class TestRAGEvaluationIntegration:
    """Integration tests for the complete RAG evaluation pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            "document": [
                "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                "Patient John Doe (DOB: 03/15/1985) was admitted with symptoms of chest pain.",
                "The financial report shows revenue of $2.5 million for Q3 2024.",
                "Deep learning uses neural networks with multiple layers to process data.",
                "Dr. Sarah Smith can be reached at (555) 123-4567 for medical consultations.",
                "Natural language processing helps computers understand human language."
            ],
            "domain": ["technology", "medical", "financial", "technology", "medical", "technology"],
            "document_id": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
            "metadata": [{"source": f"test_{i}"} for i in range(6)]
        }
        self.dataset = Dataset.from_dict(self.test_data)

    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.SentenceTransformer')
    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.KMeans')
    def test_diversity_pipeline(self, mock_kmeans, mock_sentence_transformer):
        """Test the complete diversity analysis pipeline."""
        # Mock SentenceTransformer
        mock_transformer = Mock()
        mock_transformer.encode.return_value = np.random.rand(6, 384)
        mock_sentence_transformer.return_value = mock_transformer
        
        # Mock KMeans
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 1, 2, 0, 1, 0])
        mock_clusterer.labels_ = np.array([0, 1, 2, 0, 1, 0])
        mock_kmeans.return_value = mock_clusterer
        
        # Step 1: Diversity Agent
        diversity_agent = DiversityAgentBlock(num_clusters=3)
        diversity_result = diversity_agent.generate(self.dataset)
        
        # Verify diversity agent output
        assert "diversity_cluster" in diversity_result.column_names
        assert "diversity_score" in diversity_result.column_names
        assert "semantic_features" in diversity_result.column_names
        
        # Step 2: Evaluate Diversity
        diversity_evaluator = EvaluateDiversityBlock()
        evaluation_result = diversity_evaluator.generate(diversity_result)
        
        # Verify evaluation output
        assert "diversity_evaluation_score" in evaluation_result.column_names
        assert "cluster_balance_score" in evaluation_result.column_names
        assert "semantic_spread_score" in evaluation_result.column_names
        assert "coverage_score" in evaluation_result.column_names
        assert "diversity_quality_assessment" in evaluation_result.column_names
        
        # Verify data integrity throughout pipeline
        assert len(evaluation_result) == len(self.dataset)
        
        # Verify all original columns are preserved
        for col in self.dataset.column_names:
            assert col in evaluation_result.column_names

    @patch('sdg_hub.core.blocks.rag_evaluation.privacy_agent_block.spacy.load')
    def test_privacy_pipeline(self, mock_spacy_load):
        """Test the complete privacy analysis pipeline."""
        # Mock spaCy NLP pipeline
        mock_nlp = Mock()
        mock_doc = Mock()
        
        # Mock entities for different documents
        def mock_nlp_call(text):
            mock_doc_instance = Mock()
            if "John Doe" in text:
                mock_entity1 = Mock()
                mock_entity1.text = "John Doe"
                mock_entity1.label_ = "PERSON"
                mock_entity1.start_char = 8
                mock_entity1.end_char = 16
                
                mock_entity2 = Mock()
                mock_entity2.text = "03/15/1985"
                mock_entity2.label_ = "DATE"
                mock_entity2.start_char = 23
                mock_entity2.end_char = 33
                
                mock_doc_instance.ents = [mock_entity1, mock_entity2]
            elif "Sarah Smith" in text:
                mock_entity = Mock()
                mock_entity.text = "Sarah Smith"
                mock_entity.label_ = "PERSON"
                mock_entity.start_char = 4
                mock_entity.end_char = 15
                
                mock_doc_instance.ents = [mock_entity]
            else:
                mock_doc_instance.ents = []
            
            return mock_doc_instance
        
        mock_nlp.side_effect = mock_nlp_call
        mock_spacy_load.return_value = mock_nlp
        
        # Step 1: Privacy Agent
        privacy_agent = PrivacyAgentBlock()
        privacy_result = privacy_agent.generate(self.dataset)
        
        # Verify privacy agent output
        assert "privacy_masked_document" in privacy_result.column_names
        assert "privacy_score" in privacy_result.column_names
        assert "detected_entities" in privacy_result.column_names
        
        # Step 2: Evaluate Privacy
        privacy_evaluator = EvaluatePrivacyBlock()
        evaluation_result = privacy_evaluator.generate(privacy_result)
        
        # Verify evaluation output
        assert "privacy_evaluation_score" in evaluation_result.column_names
        assert "masking_effectiveness_score" in evaluation_result.column_names
        assert "entity_coverage_score" in evaluation_result.column_names
        assert "domain_compliance_score" in evaluation_result.column_names
        assert "privacy_quality_assessment" in evaluation_result.column_names
        
        # Verify data integrity throughout pipeline
        assert len(evaluation_result) == len(self.dataset)
        
        # Verify all original columns are preserved
        for col in self.dataset.column_names:
            assert col in evaluation_result.column_names

    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.SentenceTransformer')
    @patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.KMeans')
    @patch('sdg_hub.core.blocks.rag_evaluation.privacy_agent_block.spacy.load')
    def test_complete_multi_agent_pipeline(self, mock_spacy_load, mock_kmeans, mock_sentence_transformer):
        """Test the complete multi-agent pipeline with both diversity and privacy analysis."""
        # Mock dependencies (same as individual tests)
        mock_transformer = Mock()
        mock_transformer.encode.return_value = np.random.rand(6, 384)
        mock_sentence_transformer.return_value = mock_transformer
        
        mock_clusterer = Mock()
        mock_clusterer.fit_predict.return_value = np.array([0, 1, 2, 0, 1, 0])
        mock_clusterer.labels_ = np.array([0, 1, 2, 0, 1, 0])
        mock_kmeans.return_value = mock_clusterer
        
        mock_nlp = Mock()
        mock_doc = Mock()
        mock_doc.ents = []
        mock_nlp.return_value = mock_doc
        mock_spacy_load.return_value = mock_nlp
        
        # Step 1: Diversity Analysis
        diversity_agent = DiversityAgentBlock(num_clusters=3)
        diversity_result = diversity_agent.generate(self.dataset)
        
        # Step 2: Privacy Analysis (on original dataset)
        privacy_agent = PrivacyAgentBlock()
        privacy_result = privacy_agent.generate(self.dataset)
        
        # Step 3: Combine results (simulate what the flow would do)
        # In a real flow, this would be handled by the flow orchestration
        combined_data = {}
        
        # Add all columns from both results
        for col in diversity_result.column_names:
            combined_data[col] = diversity_result[col]
        
        for col in privacy_result.column_names:
            if col not in combined_data:  # Avoid duplicating base columns
                combined_data[col] = privacy_result[col]
        
        combined_dataset = Dataset.from_dict(combined_data)
        
        # Step 4: Evaluate both aspects
        diversity_evaluator = EvaluateDiversityBlock()
        diversity_eval_result = diversity_evaluator.generate(combined_dataset)
        
        privacy_evaluator = EvaluatePrivacyBlock()
        privacy_eval_result = privacy_evaluator.generate(combined_dataset)
        
        # Verify complete pipeline results
        assert len(diversity_eval_result) == len(self.dataset)
        assert len(privacy_eval_result) == len(self.dataset)
        
        # Verify all evaluation metrics are present
        diversity_metrics = ["diversity_evaluation_score", "cluster_balance_score", 
                           "semantic_spread_score", "coverage_score"]
        privacy_metrics = ["privacy_evaluation_score", "masking_effectiveness_score",
                         "entity_coverage_score", "domain_compliance_score"]
        
        for metric in diversity_metrics:
            assert metric in diversity_eval_result.column_names
        
        for metric in privacy_metrics:
            assert metric in privacy_eval_result.column_names

    def test_data_flow_consistency(self):
        """Test that data flows consistently through the pipeline without corruption."""
        # Create a simple dataset with known values
        simple_data = {
            "document": ["Test document 1", "Test document 2"],
            "domain": ["test", "test"],
            "document_id": ["test1", "test2"],
            "metadata": [{"id": 1}, {"id": 2}]
        }
        simple_dataset = Dataset.from_dict(simple_data)
        
        # Test that each block preserves original data
        with patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.SentenceTransformer') as mock_st, \
             patch('sdg_hub.core.blocks.rag_evaluation.diversity_agent_block.KMeans') as mock_km:
            
            mock_transformer = Mock()
            mock_transformer.encode.return_value = np.random.rand(2, 384)
            mock_st.return_value = mock_transformer
            
            mock_clusterer = Mock()
            mock_clusterer.fit_predict.return_value = np.array([0, 1])
            mock_clusterer.labels_ = np.array([0, 1])
            mock_km.return_value = mock_clusterer
            
            diversity_agent = DiversityAgentBlock(num_clusters=2)
            result = diversity_agent.generate(simple_dataset)
            
            # Verify original data is preserved
            assert result["document"] == simple_data["document"]
            assert result["domain"] == simple_data["domain"]
            assert result["document_id"] == simple_data["document_id"]
            assert result["metadata"] == simple_data["metadata"]

    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Test with invalid data
        invalid_data = {
            "wrong_column": ["test"],
            "domain": ["test"]
        }
        invalid_dataset = Dataset.from_dict(invalid_data)
        
        # Each block should raise appropriate errors
        diversity_agent = DiversityAgentBlock()
        with pytest.raises(ValueError):
            diversity_agent.generate(invalid_dataset)
        
        privacy_agent = PrivacyAgentBlock()
        with pytest.raises(ValueError):
            privacy_agent.generate(invalid_dataset)
        
        diversity_evaluator = EvaluateDiversityBlock()
        with pytest.raises(ValueError):
            diversity_evaluator.generate(invalid_dataset)
        
        privacy_evaluator = EvaluatePrivacyBlock()
        with pytest.raises(ValueError):
            privacy_evaluator.generate(invalid_dataset)


if __name__ == "__main__":
    pytest.main([__file__])
