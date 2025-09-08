#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Example usage of the Multi-Agent Diverse and Private RAG Evaluation Flow.

This script demonstrates how to use the multi-agent framework for generating
diverse and privacy-preserving synthetic QA datasets for RAG evaluation.
"""

# Standard
import logging
from pathlib import Path

# Third Party
from datasets import Dataset

# Local
from sdg_hub.core.flow import Flow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_data = {
        "document": [
            """
            The patient, John Smith, was admitted to Springfield General Hospital on March 15, 2024, 
            with symptoms of chest pain and shortness of breath. Initial examination revealed elevated 
            blood pressure (160/95 mmHg) and irregular heartbeat. Dr. Sarah Johnson ordered an ECG 
            and blood tests. The patient's medical history includes diabetes mellitus type 2, diagnosed 
            in 2018, and a family history of cardiovascular disease. Treatment plan includes 
            beta-blockers and lifestyle modifications. Patient's insurance ID is MED-789456123.
            """,
            """
            Global Tech Solutions Inc. reported quarterly earnings of $2.5 million for Q3 2024, 
            representing a 15% increase from the previous quarter. The company's main revenue streams 
            include software licensing ($1.8M), consulting services ($500K), and maintenance contracts 
            ($200K). CEO Michael Davis announced plans for expansion into the European market. 
            The company's bank account (routing: 123456789, account: 987654321) shows strong liquidity. 
            Employee headcount increased to 150, with average salary of $85,000.
            """,
            """
            In the legal case Thompson vs. Acme Corporation (Case #CV-2024-001234), the plaintiff 
            alleges breach of contract regarding software delivery delays. Attorney Lisa Rodriguez 
            represents the plaintiff, while defendant is represented by Johnson & Associates. 
            The contract, signed on January 10, 2024, specified delivery within 90 days. 
            Damages sought include $500,000 in lost revenue and $100,000 in legal fees. 
            Court date is scheduled for November 20, 2024, in Superior Court of California.
            """,
            """
            The artificial intelligence research team at Stanford University published findings 
            on large language model performance in scientific domains. The study analyzed 
            15 different models across physics, chemistry, and biology tasks. Results showed 
            that specialized fine-tuning improved accuracy by 23% on average. The research 
            was funded by NSF grant #AI-2024-789 totaling $2.3 million over three years. 
            Lead researcher Dr. Amanda Chen will present findings at the NeurIPS conference.
            """
        ],
        "domain": ["medical", "financial", "legal", "academic"],
        "document_id": ["med_001", "fin_002", "leg_003", "acad_004"]
    }
    
    return Dataset.from_dict(sample_data)


def main():
    """Main execution function."""
    logger.info("Starting Multi-Agent RAG Evaluation Flow Example")
    
    # Create sample dataset
    logger.info("Creating sample dataset...")
    input_dataset = create_sample_dataset()
    logger.info(f"Created dataset with {len(input_dataset)} documents")
    
    # Load the flow
    flow_path = Path(__file__).parent / "flow.yaml"
    logger.info(f"Loading flow from: {flow_path}")
    
    try:
        flow = Flow.from_yaml(str(flow_path))
        logger.info("Flow loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load flow: {e}")
        return
    
    # Configure flow parameters
    logger.info("Configuring flow parameters...")
    flow.set_parameters({
        # Diversity Agent Parameters
        "diversity_clusters": 4,  # Smaller for demo
        "diversity_threshold": 0.6,  # Lower threshold for demo
        
        # Privacy Agent Parameters
        "privacy_domains": ["medical", "financial", "legal", "personal"],
        "masking_strategy": "entity_replacement",
        
        # QA Curation Parameters
        "qa_pairs_per_document": 2,  # Fewer pairs for demo
        "quality_threshold": 0.7,  # Lower threshold for demo
        
        # Model configuration (adjust based on your setup)
        "model": "meta-llama/Llama-3.3-70B-Instruct",  # or your preferred model
        "temperature": 0.7,
        "max_tokens": 1024,
        "async_mode": True
    })
    
    # Run the flow
    logger.info("Running multi-agent RAG evaluation flow...")
    try:
        result_dataset = flow.run(input_dataset)
        logger.info(f"Flow completed successfully! Generated {len(result_dataset)} QA pairs")
        
        # Display results summary
        logger.info("\n=== RESULTS SUMMARY ===")
        
        # Diversity statistics
        if "diversity_score" in result_dataset.column_names:
            diversity_scores = result_dataset["diversity_score"]
            avg_diversity = sum(diversity_scores) / len(diversity_scores)
            logger.info(f"Average Diversity Score: {avg_diversity:.3f}")
        
        # Privacy statistics
        if "privacy_compliance" in result_dataset.column_names:
            privacy_compliant = sum(1 for status in result_dataset["privacy_compliance"] 
                                  if status == "COMPLIANT")
            logger.info(f"Privacy Compliant QA Pairs: {privacy_compliant}/{len(result_dataset)}")
        
        # Faithfulness statistics
        if "faithfulness_judgment" in result_dataset.column_names:
            faithful_answers = sum(1 for judgment in result_dataset["faithfulness_judgment"] 
                                 if judgment == "YES")
            logger.info(f"Faithful Answers: {faithful_answers}/{len(result_dataset)}")
        
        # Sample outputs
        logger.info("\n=== SAMPLE GENERATED QA PAIRS ===")
        for i in range(min(2, len(result_dataset))):
            logger.info(f"\nQA Pair {i+1}:")
            logger.info(f"Domain: {result_dataset['domain'][i]}")
            logger.info(f"Question: {result_dataset['question'][i]}")
            logger.info(f"Answer: {result_dataset['answer'][i][:200]}...")
            if "diversity_score" in result_dataset.column_names:
                logger.info(f"Diversity Score: {result_dataset['diversity_score'][i]:.3f}")
            if "privacy_compliance" in result_dataset.column_names:
                logger.info(f"Privacy Status: {result_dataset['privacy_compliance'][i]}")
        
        # Save results (optional)
        output_path = Path("rag_evaluation_results.json")
        result_dataset.to_json(str(output_path))
        logger.info(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Flow execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
