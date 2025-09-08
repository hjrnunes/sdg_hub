# RAG Evaluation Examples

This directory contains examples and tutorials for using SDG Hub's RAG evaluation capabilities, including the multi-agent framework for generating diverse and privacy-preserving synthetic QA datasets.

## 📚 Available Examples

### 🤖 Multi-Agent Diverse and Private QA Generation
**File**: `multi_agent_diverse_private_qa_generation.ipynb`

An interactive Jupyter notebook demonstrating the complete multi-agent RAG evaluation framework based on the research paper "Diverse And Private Synthetic Datasets Generation for RAG evaluation" by Driouich et al. (2024).

### 📄 Input Schema and PDF Integration Guide
**File**: `INPUT_SCHEMA.md`

Comprehensive guide for input data format and external PDF processing integration. Includes chunking strategies, domain classification, validation examples, and best practices for large document collections.

### 🔧 Data Preprocessing for RAG Evaluation
**File**: `data_preprocessing_for_rag_eval.py`

Production-ready template for preprocessing large PDF collections for RAG evaluation. Features multiple chunking strategies (fixed-size, page-based, semantic), Docling integration, batch processing, error handling, and quality analysis tools.

**What you'll learn:**
- How to use the Diversity Agent for semantic clustering and diversity optimization
- How to apply the Privacy Agent for sensitive information detection and masking
- How to combine agents for comprehensive quality assessment
- How to visualize and analyze results
- How to prepare data for RAG evaluation dataset generation

**Prerequisites:**
```bash
pip install sentence-transformers scikit-learn spacy datasets matplotlib seaborn
python -m spacy download en_core_web_sm
```

## 🚀 Quick Start

1. **Open the notebook**: Launch Jupyter and open `multi_agent_diverse_private_qa_generation.ipynb`
2. **Install dependencies**: Run the first cell to install required packages
3. **Follow along**: Execute cells step-by-step to see the framework in action
4. **Customize**: Modify parameters and data to fit your use case

## 🎯 Framework Components

### 🔍 Diversity Agent
- **Purpose**: Maximize topical coverage and semantic variability
- **Method**: Sentence embeddings + K-means clustering
- **Output**: Diversity clusters, scores, and semantic features

### 🔒 Privacy Agent  
- **Purpose**: Detect and mask sensitive information
- **Method**: SpaCy NER + regex patterns + domain-specific detection
- **Output**: Masked documents, detected entities, privacy scores

### 🤖 QA Curation Agent
- **Purpose**: Generate diverse, privacy-compliant QA pairs
- **Method**: LLM-powered generation with multi-agent constraints
- **Output**: High-quality QA pairs suitable for RAG evaluation

## 📊 Use Cases

- **RAG System Evaluation**: Generate comprehensive test datasets
- **Privacy-Compliant AI**: Ensure sensitive data protection
- **Diverse Question Generation**: Cover multiple semantic dimensions
- **Quality Assessment**: Measure diversity, privacy, and faithfulness
- **Benchmark Creation**: Build robust evaluation datasets

## 🔧 Customization Options

### Diversity Configuration
```python
diversity_agent = DiversityAgentBlock(
    num_clusters=5,           # Adjust semantic groupings
    diversity_threshold=0.7,  # Set quality bar
    embedding_model="...",    # Choose embedding model
    clustering_method="kmeans" # Select clustering algorithm
)
```

### Privacy Configuration
```python
privacy_agent = PrivacyAgentBlock(
    privacy_domains=["medical", "financial", "legal"],
    masking_strategy="entity_replacement",  # or "redaction", "synthetic_replacement"
    entity_types=["PERSON", "ORG", "MONEY", "PHONE", "EMAIL"],
    preserve_structure=True
)
```

## 📈 Expected Outputs

After running the notebook, you'll have:
- **Analyzed Documents**: With diversity and privacy scores
- **Semantic Clusters**: Grouped by topical similarity  
- **Privacy-Masked Content**: Safe for QA generation
- **Quality Metrics**: Comprehensive assessment dashboard
- **Visualization**: Charts showing analysis results

## 🔗 Related Resources

- **Research Paper**: [Diverse And Private Synthetic Datasets Generation for RAG evaluation](https://arxiv.org/pdf/2508.18929)
- **Flow Configuration**: `../../src/sdg_hub/flows/rag_evaluation/multi_agent_diverse_private/`
- **Block Documentation**: `../../docs/blocks/`
- **SDG Hub Core**: `../../src/sdg_hub/core/`

## 🤝 Contributing

Found an issue or want to add more examples? Please:
1. Open an issue describing the problem or enhancement
2. Submit a pull request with your improvements
3. Follow the existing code style and documentation format

## 📄 License

This example is part of SDG Hub and is licensed under Apache-2.0.
