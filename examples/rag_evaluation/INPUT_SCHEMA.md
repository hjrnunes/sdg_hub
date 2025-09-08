# Input Schema for Multi-Agent RAG Evaluation Flow

This document describes the input data requirements for the Multi-Agent Diverse and Private Synthetic QA Dataset Generation flow.

## 📋 Required Input Schema

### Core Required Columns

```python
{
    "document": str,     # Main document text content
    "domain": str        # Domain classification for the document
}
```

### Optional Columns

```python
{
    "document_id": str,      # Unique identifier for the document
    "metadata": dict         # Additional document metadata
}
```

## 📊 Example Dataset Structure

```python
from datasets import Dataset

# Minimal required structure
data = {
    "document": [
        "Your document text content here...",
        "Another document with different content...",
    ],
    "domain": [
        "medical",      # Domain classification
        "financial"     # Supported: medical, financial, legal, academic, technology, etc.
    ]
}

# With optional fields
enhanced_data = {
    "document": ["Document content..."],
    "domain": ["medical"],
    "document_id": ["doc_001"],
    "metadata": [{"source": "research_paper.pdf", "pages": 15, "author": "Dr. Smith"}]
}

dataset = Dataset.from_dict(data)
```

## 🔧 Data Requirements

### Document Content
- **Minimum length**: 100 words recommended for meaningful analysis
- **Maximum length**: No hard limit, but very long documents may be truncated by LLM context windows
- **Format**: Plain text (HTML/Markdown will be processed as-is)
- **Language**: English (current NER models optimized for English)

### Domain Classification
Supported domains for privacy-aware processing:
- `"medical"` - Healthcare, patient data, medical research
- `"financial"` - Banking, investments, financial reports
- `"legal"` - Legal documents, contracts, court cases
- `"academic"` - Research papers, educational content
- `"technology"` - Technical documentation, software
- `"personal"` - Personal information, private communications
- `"environmental"` - Climate data, environmental reports
- `"general"` - Generic content not fitting other categories

## 📄 PDF Processing Integration Guide

### For Large PDF Collections

If you're processing large PDFs with Docling or similar tools, here's how to prepare data for this flow:

#### 1. PDF Processing Pipeline with Chunking (External)

```python
# Example external processing script with chunking support
import docling
from datasets import Dataset
import json
from pathlib import Path

def process_pdfs_to_schema(pdf_directory: str, 
                          chunk_strategy: str = "none",
                          chunk_size: int = 2000,
                          chunk_overlap: int = 200) -> Dataset:
    """
    Process PDFs using Docling with chunking support.
    This code should live in your external repository.
    
    Args:
        pdf_directory: Path to PDF files
        chunk_strategy: "none", "fixed_size", "page_based", or "semantic"
        chunk_size: Size of chunks in characters (for fixed_size)
        chunk_overlap: Overlap between chunks in characters
    """
    documents = []
    domains = []
    document_ids = []
    metadata = []
    
    for pdf_file in Path(pdf_directory).glob("*.pdf"):
        # Process with Docling
        doc = docling.load_document(pdf_file)
        
        # Apply chunking strategy
        if chunk_strategy == "page_based":
            # Extract page by page
            for i, page in enumerate(doc.pages):
                page_text = page.export_to_text()
                if len(page_text.strip()) > 50:  # Skip empty pages
                    domain = classify_document_domain(page_text)
                    documents.append(page_text)
                    domains.append(domain)
                    document_ids.append(f"{pdf_file.stem}_page_{i+1}")
                    metadata.append({
                        "source_file": str(pdf_file),
                        "page_number": i + 1,
                        "chunk_strategy": "page_based",
                        "processing_date": datetime.now().isoformat(),
                        "docling_version": docling.__version__
                    })
        
        elif chunk_strategy == "semantic":
            # Extract structured content for semantic chunking
            structured_content = doc.export_to_dict()
            sections = extract_sections(structured_content)
            
            for i, (section_text, section_title) in enumerate(sections):
                if len(section_text.strip()) > 100:
                    domain = classify_document_domain(section_text)
                    documents.append(section_text)
                    domains.append(domain)
                    document_ids.append(f"{pdf_file.stem}_section_{i}")
                    metadata.append({
                        "source_file": str(pdf_file),
                        "section_title": section_title,
                        "chunk_strategy": "semantic",
                        "processing_date": datetime.now().isoformat(),
                        "docling_version": docling.__version__
                    })
        
        elif chunk_strategy == "fixed_size":
            # Extract full text and chunk by size
            text_content = doc.export_to_text()
            domain = classify_document_domain(text_content)
            
            chunks = create_fixed_size_chunks(text_content, chunk_size, chunk_overlap)
            for i, chunk_text in enumerate(chunks):
                documents.append(chunk_text)
                domains.append(domain)
                document_ids.append(f"{pdf_file.stem}_chunk_{i}")
                metadata.append({
                    "source_file": str(pdf_file),
                    "chunk_strategy": "fixed_size",
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processing_date": datetime.now().isoformat(),
                    "docling_version": docling.__version__
                })
        
        else:  # No chunking
            text_content = doc.export_to_text()
            domain = classify_document_domain(text_content)
            documents.append(text_content)
            domains.append(domain)
            document_ids.append(pdf_file.stem)
            metadata.append({
                "source_file": str(pdf_file),
                "pages": len(doc.pages),
                "chunk_strategy": "none",
                "processing_date": datetime.now().isoformat(),
                "docling_version": docling.__version__
            })
    
    # Create dataset in SDG Hub format
    return Dataset.from_dict({
        "document": documents,
        "domain": domains,
        "document_id": document_ids,
        "metadata": metadata
    })

# Usage examples
dataset_whole = process_pdfs_to_schema("path/to/pdfs/", chunk_strategy="none")
dataset_chunks = process_pdfs_to_schema("path/to/pdfs/", chunk_strategy="fixed_size", chunk_size=1500)
dataset_pages = process_pdfs_to_schema("path/to/pdfs/", chunk_strategy="page_based")
dataset_semantic = process_pdfs_to_schema("path/to/pdfs/", chunk_strategy="semantic")
```

#### 2. Domain Classification Helper

```python
def classify_document_domain(text: str) -> str:
    """
    Classify document domain based on content.
    Implement your own logic or use a classifier.
    """
    text_lower = text.lower()
    
    # Simple keyword-based classification
    if any(word in text_lower for word in ["patient", "medical", "diagnosis", "treatment"]):
        return "medical"
    elif any(word in text_lower for word in ["financial", "revenue", "profit", "investment"]):
        return "financial"
    elif any(word in text_lower for word in ["legal", "contract", "court", "lawsuit"]):
        return "legal"
    elif any(word in text_lower for word in ["research", "study", "university", "academic"]):
        return "academic"
    elif any(word in text_lower for word in ["technology", "software", "algorithm", "system"]):
        return "technology"
    else:
        return "general"
```

#### 3. Loading Processed Data

```python
# In your SDG Hub workflow
from datasets import Dataset

# Load your processed dataset
dataset = Dataset.load_from_disk("processed_documents")

# Or load from JSON/CSV
dataset = Dataset.from_json("processed_documents.json")
dataset = Dataset.from_csv("processed_documents.csv")

# Verify schema compliance
required_columns = ["document", "domain"]
assert all(col in dataset.column_names for col in required_columns), \
    f"Missing required columns. Found: {dataset.column_names}"

print(f"✅ Dataset loaded: {len(dataset)} documents")
print(f"📊 Domains: {set(dataset['domain'])}")
```

## 🔄 Integration Workflow

### Recommended External Repository Structure

```
your-pdf-processing-repo/
├── src/
│   ├── pdf_processor.py          # Docling integration
│   ├── domain_classifier.py      # Domain classification
│   └── schema_converter.py       # Convert to SDG Hub format
├── config/
│   ├── domains.yaml              # Domain classification rules
│   └── processing_config.yaml    # Processing parameters
├── data/
│   ├── input/                    # Raw PDFs
│   ├── processed/                # Processed datasets
│   └── output/                   # Final SDG Hub format
├── notebooks/
│   └── pdf_processing_demo.ipynb # Processing examples
└── requirements.txt              # Dependencies (docling, etc.)
```

### Integration Steps

1. **Process PDFs** (in your external repo):
   ```bash
   python src/pdf_processor.py --input data/input/ --output data/processed/
   ```

2. **Convert to SDG Hub format**:
   ```bash
   python src/schema_converter.py --input data/processed/ --output data/output/
   ```

3. **Run SDG Hub flow**:
   ```python
   from sdg_hub.core.flow import Flow
   
   # Load your processed data
   dataset = Dataset.load_from_disk("data/output/sdg_hub_format")
   
   # Run multi-agent flow
   flow = Flow.from_yaml("flows/rag_evaluation/multi_agent_diverse_private/flow.yaml")
   result = flow.run(dataset)
   ```

## 📝 Data Validation

### Pre-processing Validation

```python
def validate_dataset_schema(dataset: Dataset) -> bool:
    """Validate dataset meets SDG Hub requirements."""
    
    # Check required columns
    required_cols = ["document", "domain"]
    missing_cols = [col for col in required_cols if col not in dataset.column_names]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check data types
    for i, doc in enumerate(dataset["document"]):
        if not isinstance(doc, str) or len(doc.strip()) < 50:
            print(f"Warning: Document {i} may be too short ({len(doc)} chars)")
    
    # Check domain values
    valid_domains = {"medical", "financial", "legal", "academic", "technology", "personal", "environmental", "general"}
    invalid_domains = set(dataset["domain"]) - valid_domains
    if invalid_domains:
        print(f"Warning: Unknown domains found: {invalid_domains}")
    
    print(f"✅ Dataset validation passed: {len(dataset)} documents")
    return True

# Usage
validate_dataset_schema(your_dataset)
```

## 📄 Chunking Strategies for Large Documents

When processing large PDFs, chunking is essential for optimal performance with the multi-agent framework. Here are the available strategies:

### 1. **No Chunking** (`chunk_strategy="none"`)
- **Use when**: Documents are already appropriately sized (<5k words)
- **Pros**: Preserves full document context
- **Cons**: May exceed LLM context limits for very large documents

### 2. **Fixed-Size Chunking** (`chunk_strategy="fixed_size"`)
- **Use when**: You need consistent chunk sizes for processing
- **Parameters**: 
  - `chunk_size`: Characters per chunk (recommended: 1500-3000)
  - `chunk_overlap`: Overlap between chunks (recommended: 10-20% of chunk_size)
- **Pros**: Predictable chunk sizes, good for batch processing
- **Cons**: May split sentences or concepts unnaturally

### 3. **Page-Based Chunking** (`chunk_strategy="page_based"`)
- **Use when**: Document structure aligns with pages (reports, papers)
- **Pros**: Preserves natural document boundaries
- **Cons**: Inconsistent chunk sizes, may have very short or long pages

### 4. **Semantic Chunking** (`chunk_strategy="semantic"`)
- **Use when**: Documents have clear structure (sections, headings)
- **Pros**: Preserves semantic coherence, natural boundaries
- **Cons**: Requires good document structure, variable chunk sizes

### Chunking Recommendations by Document Type

| Document Type | Recommended Strategy | Chunk Size | Notes |
|---------------|---------------------|------------|-------|
| Research Papers | `semantic` | N/A | Use sections/headings |
| Legal Documents | `semantic` or `page_based` | N/A | Preserve legal structure |
| Technical Manuals | `semantic` | N/A | Use chapter/section breaks |
| Reports | `page_based` or `fixed_size` | 2000-3000 | Depends on structure |
| Books | `semantic` or `fixed_size` | 2500-4000 | Use chapters if available |
| Articles | `fixed_size` | 1500-2500 | Usually no clear structure |

### Example Chunking Configuration

```python
# For research papers with clear sections
converter = PDFToSDGHubConverter(
    chunk_strategy="semantic",
    output_format="huggingface"
)

# For mixed document types
converter = PDFToSDGHubConverter(
    chunk_strategy="fixed_size",
    chunk_size=2000,
    chunk_overlap=300,
    output_format="huggingface"
)

# For page-structured reports
converter = PDFToSDGHubConverter(
    chunk_strategy="page_based",
    output_format="huggingface"
)
```

## 🎯 Best Practices

### For Large Document Collections

1. **Choose Appropriate Chunking**: Select strategy based on document structure and content type
2. **Batch Processing**: Process PDFs in batches to manage memory usage
3. **Quality Filtering**: Remove documents with poor OCR quality or insufficient content
4. **Domain Consistency**: Ensure consistent domain labeling across your collection
5. **Chunk Size Optimization**: Test different chunk sizes for your specific use case

### For Privacy-Sensitive Content

1. **Pre-screening**: Remove obviously sensitive documents before processing
2. **Domain-Specific Rules**: Customize privacy detection for your specific domain
3. **Validation**: Always review privacy masking results before generating QA pairs

### For Production Use

1. **Version Control**: Track processing versions and parameters
2. **Metadata Preservation**: Keep rich metadata for traceability
3. **Quality Metrics**: Monitor document quality and processing success rates
4. **Incremental Processing**: Support adding new documents to existing datasets

## 🔗 Related Resources

- **Docling Documentation**: [Docling GitHub](https://github.com/DS4SD/docling)
- **SDG Hub Flow**: `flow.yaml` in this directory
- **Example Notebook**: `../../examples/rag_evaluation/multi_agent_diverse_private_qa_generation.ipynb`
- **Privacy Configuration**: `privacy_agent_block.py` for customizing entity detection

## ❓ FAQ

**Q: Can I use documents in languages other than English?**
A: The current privacy detection uses English NER models. For other languages, you'd need to configure different SpaCy models.

**Q: How do I handle documents with mixed content types?**
A: Use the most appropriate domain classification, or split documents by section if they contain distinctly different content types.

**Q: What if my PDFs have poor text extraction quality?**
A: Consider using OCR post-processing or document quality filtering before feeding into the flow.

**Q: Can I add custom privacy entity types?**
A: Yes! Modify the `entity_types` parameter in the PrivacyAgentBlock configuration to include your custom entities.
