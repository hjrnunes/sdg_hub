#!/usr/bin/env python3
"""
Data Preprocessing for RAG Evaluation Flow

This script demonstrates how to preprocess large PDF collections for the 
SDG Hub multi-agent RAG evaluation framework. It includes comprehensive 
chunking strategies, quality analysis, and seamless integration with Docling.

This code is meant to live in your external repository, separate from SDG Hub.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Third-party imports (install in your external repo)
from datasets import Dataset
import docling  # Your PDF processing library
import pandas as pd


class RAGEvalDataPreprocessor:
    """Preprocess PDF collections for SDG Hub RAG evaluation flow."""
    
    def __init__(self, 
                 output_format: str = "huggingface",
                 domain_classifier: Optional[callable] = None,
                 chunk_strategy: str = "none",
                 chunk_size: int = 2000,
                 chunk_overlap: int = 200):
        """
        Initialize the converter.
        
        Args:
            output_format: Format for output ("huggingface", "json", "csv")
            domain_classifier: Custom domain classification function
            chunk_strategy: Chunking strategy ("none", "fixed_size", "semantic", "page_based")
            chunk_size: Size of chunks in characters (for fixed_size strategy)
            chunk_overlap: Overlap between chunks in characters
        """
        self.output_format = output_format
        self.domain_classifier = domain_classifier or self._default_domain_classifier
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_pdf_directory(self, 
                             pdf_directory: Path, 
                             output_path: Path,
                             batch_size: int = 10) -> Dataset:
        """
        Process all PDFs in a directory and convert to SDG Hub format.
        
        Args:
            pdf_directory: Path to directory containing PDFs
            output_path: Path to save processed dataset
            batch_size: Number of PDFs to process at once
            
        Returns:
            Dataset in SDG Hub format
        """
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        all_domains = []
        all_document_ids = []
        all_metadata = []
        
        # Process in batches
        for i in range(0, len(pdf_files), batch_size):
            batch_files = pdf_files[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(pdf_files)-1)//batch_size + 1}")
            
            for pdf_file in batch_files:
                try:
                    chunks = self._process_single_pdf(pdf_file)
                    if chunks:
                        # Add all chunks from this PDF
                        for doc, domain, doc_id, metadata in chunks:
                            all_documents.append(doc)
                            all_domains.append(domain)
                            all_document_ids.append(doc_id)
                            all_metadata.append(metadata)
                        
                        print(f"  ✅ {pdf_file.name}: {len(chunks)} chunks created")
                except Exception as e:
                    print(f"  ❌ Error processing {pdf_file}: {e}")
                    continue
        
        # Create dataset
        dataset_dict = {
            "document": all_documents,
            "domain": all_domains,
            "document_id": all_document_ids,
            "metadata": all_metadata
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Save dataset
        self._save_dataset(dataset, output_path)
        
        print(f"✅ Processed {len(dataset)} documents")
        print(f"📊 Domains: {set(dataset['domain'])}")
        
        return dataset
    
    def _process_single_pdf(self, pdf_file: Path) -> Optional[List[tuple]]:
        """Process a single PDF file and return list of chunks."""
        try:
            # Load and process with Docling
            doc = docling.load_document(str(pdf_file))
            
            # Extract text content and structure
            if self.chunk_strategy == "page_based":
                # Extract page-by-page for page-based chunking
                page_contents = []
                for i, page in enumerate(doc.pages):
                    page_text = page.export_to_text()
                    if page_text.strip():
                        page_contents.append((page_text, i + 1))
                
                if not page_contents:
                    print(f"Skipping {pdf_file.name}: no readable pages")
                    return None
                    
                # Create chunks from pages
                chunks = self._create_page_based_chunks(page_contents, pdf_file)
                
            elif self.chunk_strategy == "semantic":
                # Extract with structure for semantic chunking
                structured_content = doc.export_to_dict()  # Get structured content
                chunks = self._create_semantic_chunks(structured_content, pdf_file)
                
            else:
                # Extract full text for fixed-size or no chunking
                text_content = doc.export_to_text()
                
                # Quality check
                if len(text_content.strip()) < 100:
                    print(f"Skipping {pdf_file.name}: content too short")
                    return None
                
                if self.chunk_strategy == "fixed_size":
                    chunks = self._create_fixed_size_chunks(text_content, pdf_file)
                else:
                    # No chunking - return single document
                    chunks = self._create_single_document(text_content, pdf_file)
            
            return chunks
            
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
            return None
    
    def _create_single_document(self, text_content: str, pdf_file: Path) -> List[tuple]:
        """Create a single document (no chunking)."""
        domain = self.domain_classifier(text_content)
        doc_id = pdf_file.stem
        
        metadata = {
            "source_file": str(pdf_file),
            "file_size_mb": pdf_file.stat().st_size / (1024 * 1024),
            "processing_date": datetime.now().isoformat(),
            "docling_version": getattr(docling, '__version__', 'unknown'),
            "text_length": len(text_content),
            "chunk_strategy": "none",
            "chunk_index": 0,
            "total_chunks": 1
        }
        
        return [(text_content, domain, doc_id, metadata)]
    
    def _create_fixed_size_chunks(self, text_content: str, pdf_file: Path) -> List[tuple]:
        """Create fixed-size chunks with overlap."""
        chunks = []
        domain = self.domain_classifier(text_content)  # Classify once for the whole document
        
        # Split into sentences for better chunk boundaries
        sentences = text_content.split('. ')
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 2 > self.chunk_size and current_chunk:
                # Save current chunk
                doc_id = f"{pdf_file.stem}_chunk_{chunk_index}"
                
                metadata = {
                    "source_file": str(pdf_file),
                    "file_size_mb": pdf_file.stat().st_size / (1024 * 1024),
                    "processing_date": datetime.now().isoformat(),
                    "docling_version": getattr(docling, '__version__', 'unknown'),
                    "text_length": len(current_chunk),
                    "chunk_strategy": "fixed_size",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "chunk_index": chunk_index,
                    "total_chunks": "TBD"  # Will be updated later
                }
                
                chunks.append((current_chunk.strip(), domain, doc_id, metadata))
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + ". " + sentence
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk += ". " + sentence if current_chunk else sentence
        
        # Add final chunk if it has content
        if current_chunk.strip():
            doc_id = f"{pdf_file.stem}_chunk_{chunk_index}"
            
            metadata = {
                "source_file": str(pdf_file),
                "file_size_mb": pdf_file.stat().st_size / (1024 * 1024),
                "processing_date": datetime.now().isoformat(),
                "docling_version": getattr(docling, '__version__', 'unknown'),
                "text_length": len(current_chunk),
                "chunk_strategy": "fixed_size",
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunk_index": chunk_index,
                "total_chunks": chunk_index + 1
            }
            
            chunks.append((current_chunk.strip(), domain, doc_id, metadata))
        
        # Update total_chunks in all metadata
        total_chunks = len(chunks)
        for i, (text, domain, doc_id, metadata) in enumerate(chunks):
            metadata["total_chunks"] = total_chunks
            chunks[i] = (text, domain, doc_id, metadata)
        
        return chunks
    
    def _create_page_based_chunks(self, page_contents: List[tuple], pdf_file: Path) -> List[tuple]:
        """Create chunks based on PDF pages."""
        chunks = []
        
        for page_text, page_num in page_contents:
            if len(page_text.strip()) < 50:  # Skip very short pages
                continue
                
            domain = self.domain_classifier(page_text)
            doc_id = f"{pdf_file.stem}_page_{page_num}"
            
            metadata = {
                "source_file": str(pdf_file),
                "file_size_mb": pdf_file.stat().st_size / (1024 * 1024),
                "processing_date": datetime.now().isoformat(),
                "docling_version": getattr(docling, '__version__', 'unknown'),
                "text_length": len(page_text),
                "chunk_strategy": "page_based",
                "page_number": page_num,
                "chunk_index": len(chunks),
                "total_chunks": len(page_contents)
            }
            
            chunks.append((page_text.strip(), domain, doc_id, metadata))
        
        return chunks
    
    def _create_semantic_chunks(self, structured_content: Dict, pdf_file: Path) -> List[tuple]:
        """Create semantic chunks based on document structure (sections, headings, etc.)."""
        chunks = []
        
        # This is a simplified example - you would implement more sophisticated
        # semantic chunking based on Docling's structured output
        
        def extract_sections(content, parent_title=""):
            """Recursively extract sections from structured content."""
            sections = []
            
            if isinstance(content, dict):
                # Handle different content types from Docling
                if "sections" in content:
                    for section in content["sections"]:
                        title = section.get("title", "")
                        text = section.get("text", "")
                        
                        if text and len(text.strip()) > 100:
                            full_title = f"{parent_title} - {title}" if parent_title else title
                            sections.append((text.strip(), full_title))
                        
                        # Recursively process subsections
                        subsections = extract_sections(section, title)
                        sections.extend(subsections)
                
                elif "text" in content and len(content["text"].strip()) > 100:
                    sections.append((content["text"].strip(), parent_title))
            
            return sections
        
        sections = extract_sections(structured_content)
        
        if not sections:
            # Fallback to full text if no sections found
            full_text = str(structured_content)
            if len(full_text.strip()) > 100:
                sections = [(full_text.strip(), "Full Document")]
        
        for i, (section_text, section_title) in enumerate(sections):
            domain = self.domain_classifier(section_text)
            doc_id = f"{pdf_file.stem}_section_{i}"
            
            metadata = {
                "source_file": str(pdf_file),
                "file_size_mb": pdf_file.stat().st_size / (1024 * 1024),
                "processing_date": datetime.now().isoformat(),
                "docling_version": getattr(docling, '__version__', 'unknown'),
                "text_length": len(section_text),
                "chunk_strategy": "semantic",
                "section_title": section_title,
                "chunk_index": i,
                "total_chunks": len(sections)
            }
            
            chunks.append((section_text, domain, doc_id, metadata))
        
        return chunks
    
    def _default_domain_classifier(self, text: str) -> str:
        """
        Simple keyword-based domain classification.
        Replace with your own sophisticated classifier.
        """
        text_lower = text.lower()
        
        # Medical domain
        medical_keywords = [
            "patient", "medical", "diagnosis", "treatment", "hospital", 
            "doctor", "physician", "clinical", "healthcare", "medicine",
            "symptoms", "therapy", "prescription", "disease", "health"
        ]
        
        # Financial domain
        financial_keywords = [
            "financial", "revenue", "profit", "investment", "banking",
            "money", "dollar", "earnings", "budget", "cost", "price",
            "market", "stock", "bond", "fund", "loan", "credit"
        ]
        
        # Legal domain
        legal_keywords = [
            "legal", "contract", "court", "lawsuit", "attorney", "lawyer",
            "judge", "law", "regulation", "compliance", "agreement",
            "litigation", "case", "defendant", "plaintiff", "verdict"
        ]
        
        # Academic domain
        academic_keywords = [
            "research", "study", "university", "academic", "scholar",
            "paper", "journal", "experiment", "analysis", "methodology",
            "hypothesis", "conclusion", "literature", "citation", "peer"
        ]
        
        # Technology domain
        technology_keywords = [
            "technology", "software", "algorithm", "system", "computer",
            "data", "digital", "programming", "development", "technical",
            "platform", "application", "database", "network", "security"
        ]
        
        # Count keyword matches
        domain_scores = {
            "medical": sum(1 for kw in medical_keywords if kw in text_lower),
            "financial": sum(1 for kw in financial_keywords if kw in text_lower),
            "legal": sum(1 for kw in legal_keywords if kw in text_lower),
            "academic": sum(1 for kw in academic_keywords if kw in text_lower),
            "technology": sum(1 for kw in technology_keywords if kw in text_lower)
        }
        
        # Return domain with highest score, or "general" if no clear match
        max_domain = max(domain_scores, key=domain_scores.get)
        return max_domain if domain_scores[max_domain] > 2 else "general"
    
    def _save_dataset(self, dataset: Dataset, output_path: Path):
        """Save dataset in specified format."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == "huggingface":
            dataset.save_to_disk(output_path / "sdg_hub_dataset")
        elif self.output_format == "json":
            dataset.to_json(output_path / "sdg_hub_dataset.json")
        elif self.output_format == "csv":
            # Note: metadata column will be JSON strings in CSV
            df = dataset.to_pandas()
            df['metadata'] = df['metadata'].apply(json.dumps)
            df.to_csv(output_path / "sdg_hub_dataset.csv", index=False)
        
        print(f"💾 Dataset saved to {output_path}")


def validate_sdg_hub_dataset(dataset: Dataset) -> bool:
    """Validate that dataset meets SDG Hub requirements."""
    
    # Check required columns
    required_columns = ["document", "domain"]
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Check data quality
    issues = []
    
    for i, (doc, domain) in enumerate(zip(dataset["document"], dataset["domain"])):
        if not isinstance(doc, str) or len(doc.strip()) < 50:
            issues.append(f"Document {i}: too short ({len(doc)} chars)")
        
        if not isinstance(domain, str) or not domain.strip():
            issues.append(f"Document {i}: invalid domain '{domain}'")
    
    if issues:
        print(f"⚠️ Data quality issues found:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more issues")
    
    # Summary
    valid_domains = {"medical", "financial", "legal", "academic", "technology", "personal", "environmental", "general"}
    unknown_domains = set(dataset["domain"]) - valid_domains
    
    print(f"📊 Dataset Validation Summary:")
    print(f"  ✅ Total documents: {len(dataset)}")
    print(f"  ✅ Required columns present: {all(col in dataset.column_names for col in required_columns)}")
    print(f"  📋 Domains found: {set(dataset['domain'])}")
    
    if unknown_domains:
        print(f"  ⚠️ Unknown domains (will work but may have suboptimal privacy detection): {unknown_domains}")
    
    return len(issues) == 0


def main():
    """Example usage of the PDF to SDG Hub converter with different chunking strategies."""
    
    # Configuration
    pdf_directory = Path("data/input_pdfs")  # Your PDF directory
    output_directory = Path("data/sdg_hub_format")  # Output directory
    
    print("🔧 RAG Evaluation Data Preprocessor - Chunking Examples")
    print("=" * 60)
    
    # Example 1: No chunking (whole documents)
    print("\n📄 Example 1: No Chunking (Whole Documents)")
    converter_whole = RAGEvalDataPreprocessor(
        output_format="huggingface",
        chunk_strategy="none"
    )
    
    # Example 2: Fixed-size chunking
    print("\n✂️ Example 2: Fixed-Size Chunking")
    converter_fixed = RAGEvalDataPreprocessor(
        output_format="huggingface",
        chunk_strategy="fixed_size",
        chunk_size=2000,      # 2000 characters per chunk
        chunk_overlap=200     # 200 character overlap
    )
    
    # Example 3: Page-based chunking
    print("\n📃 Example 3: Page-Based Chunking")
    converter_pages = RAGEvalDataPreprocessor(
        output_format="huggingface",
        chunk_strategy="page_based"
    )
    
    # Example 4: Semantic chunking (using Docling structure)
    print("\n🧠 Example 4: Semantic Chunking")
    converter_semantic = RAGEvalDataPreprocessor(
        output_format="huggingface",
        chunk_strategy="semantic"
    )
    
    # Choose which strategy to run
    strategies = {
        "1": ("whole", converter_whole),
        "2": ("fixed", converter_fixed),
        "3": ("pages", converter_pages),
        "4": ("semantic", converter_semantic)
    }
    
    print("\nSelect chunking strategy:")
    print("1. No chunking (whole documents)")
    print("2. Fixed-size chunks (2000 chars, 200 overlap)")
    print("3. Page-based chunks")
    print("4. Semantic chunks (sections/headings)")
    
    # For demo purposes, let's run fixed-size chunking
    strategy_name, converter = strategies["2"]
    
    # Process PDFs
    if pdf_directory.exists():
        print(f"\n🚀 Processing PDFs with {strategy_name} strategy...")
        
        dataset = converter.process_pdf_directory(
            pdf_directory=pdf_directory,
            output_path=output_directory / strategy_name,
            batch_size=5
        )
        
        # Show chunking statistics
        print(f"\n📊 Chunking Statistics:")
        if "chunk_strategy" in dataset["metadata"][0]:
            chunk_info = {}
            for metadata in dataset["metadata"]:
                strategy = metadata.get("chunk_strategy", "unknown")
                if strategy not in chunk_info:
                    chunk_info[strategy] = {"count": 0, "total_chunks": 0}
                chunk_info[strategy]["count"] += 1
                chunk_info[strategy]["total_chunks"] = metadata.get("total_chunks", 1)
            
            for strategy, info in chunk_info.items():
                print(f"  • Strategy: {strategy}")
                print(f"    - Total chunks: {info['count']}")
                print(f"    - Avg chunks per doc: {info['count'] / len(set(m.get('source_file', '') for m in dataset['metadata'])):.1f}")
        
        # Validate result
        validate_sdg_hub_dataset(dataset)
        
        print(f"\n🚀 Ready for SDG Hub Multi-Agent Flow!")
        print("Next steps:")
        print("1. Load the dataset in your SDG Hub workflow:")
        print(f"   dataset = Dataset.load_from_disk('{output_directory}/{strategy_name}/sdg_hub_dataset')")
        print("2. Run the multi-agent flow:")
        print("   flow = Flow.from_yaml('flows/rag_evaluation/multi_agent_diverse_private/flow.yaml')")
        print("   result = flow.run(dataset)")
        
    else:
        print(f"❌ PDF directory not found: {pdf_directory}")
        print("Please create the directory and add your PDF files.")
        
        # Show example of how to create test data
        print("\n💡 To test with sample data:")
        print("mkdir -p data/input_pdfs")
        print("# Add your PDF files to data/input_pdfs/")


def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies with sample text."""
    
    sample_text = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn from data. This field has grown tremendously in recent years.
    
    Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised 
    learning, and reinforcement learning. Each has its own applications and methods.
    
    Supervised Learning
    
    In supervised learning, algorithms learn from labeled training data. Common examples 
    include classification and regression tasks. Popular algorithms include linear 
    regression, decision trees, and neural networks.
    
    Conclusion
    
    Machine learning continues to evolve and find new applications across industries.
    The future holds exciting possibilities for this technology.
    """
    
    print("\n🔍 Chunking Strategy Demonstration")
    print("=" * 50)
    
    # Create a mock PDF file for demonstration
    class MockPDFFile:
        def __init__(self, name):
            self.name = name
            self.stem = "sample_doc"
        
        def stat(self):
            class MockStat:
                st_size = 1024 * 10  # 10KB
            return MockStat()
    
    mock_file = MockPDFFile("sample_doc.pdf")
    
    # Test different strategies
    converter = RAGEvalDataPreprocessor()
    
    print("\n1️⃣ Fixed-Size Chunking (500 chars, 50 overlap):")
    converter.chunk_strategy = "fixed_size"
    converter.chunk_size = 500
    converter.chunk_overlap = 50
    
    chunks = converter._create_fixed_size_chunks(sample_text, mock_file)
    for i, (text, domain, doc_id, metadata) in enumerate(chunks):
        print(f"   Chunk {i}: {len(text)} chars - {text[:100]}...")
    
    print(f"\n   📊 Created {len(chunks)} chunks")
    
    print("\n2️⃣ No Chunking:")
    converter.chunk_strategy = "none"
    whole_doc = converter._create_single_document(sample_text, mock_file)
    print(f"   Single document: {len(whole_doc[0][0])} chars")
    
    print("\n✅ Chunking demonstration complete!")


# Additional utility functions for chunking
def analyze_chunk_quality(dataset: Dataset) -> Dict[str, Any]:
    """Analyze the quality and characteristics of chunks in a dataset."""
    
    analysis = {
        "total_chunks": len(dataset),
        "chunk_strategies": {},
        "size_distribution": {
            "min_size": float('inf'),
            "max_size": 0,
            "avg_size": 0,
            "sizes": []
        },
        "domain_distribution": {},
        "source_files": set()
    }
    
    for i in range(len(dataset)):
        metadata = dataset["metadata"][i]
        text_length = len(dataset["document"][i])
        domain = dataset["domain"][i]
        
        # Track chunk strategies
        strategy = metadata.get("chunk_strategy", "unknown")
        if strategy not in analysis["chunk_strategies"]:
            analysis["chunk_strategies"][strategy] = 0
        analysis["chunk_strategies"][strategy] += 1
        
        # Track size distribution
        analysis["size_distribution"]["sizes"].append(text_length)
        analysis["size_distribution"]["min_size"] = min(analysis["size_distribution"]["min_size"], text_length)
        analysis["size_distribution"]["max_size"] = max(analysis["size_distribution"]["max_size"], text_length)
        
        # Track domain distribution
        if domain not in analysis["domain_distribution"]:
            analysis["domain_distribution"][domain] = 0
        analysis["domain_distribution"][domain] += 1
        
        # Track source files
        source_file = metadata.get("source_file", "unknown")
        analysis["source_files"].add(source_file)
    
    # Calculate average size
    if analysis["size_distribution"]["sizes"]:
        analysis["size_distribution"]["avg_size"] = sum(analysis["size_distribution"]["sizes"]) / len(analysis["size_distribution"]["sizes"])
    
    analysis["unique_source_files"] = len(analysis["source_files"])
    
    return analysis


if __name__ == "__main__":
    main()


# Example usage in a separate script or notebook:
"""
# In your external repository:

from pdf_to_sdg_hub import PDFToSDGHubConverter, validate_sdg_hub_dataset
from datasets import Dataset

# Process your PDFs
converter = PDFToSDGHubConverter()
dataset = converter.process_pdf_directory("my_pdfs/", "output/")

# Validate
validate_sdg_hub_dataset(dataset)

# Use with SDG Hub (in your SDG Hub environment)
from sdg_hub.core.flow import Flow

flow = Flow.from_yaml("flows/rag_evaluation/multi_agent_diverse_private/flow.yaml")
result = flow.run(dataset)
"""
