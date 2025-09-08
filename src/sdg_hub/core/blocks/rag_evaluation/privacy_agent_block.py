# SPDX-License-Identifier: Apache-2.0
"""Privacy Agent Block for sensitive information detection and masking.

This module implements the Privacy Agent from the multi-agent RAG evaluation framework,
which detects and masks sensitive information across multiple domains to ensure
privacy-preserving synthetic dataset generation.
"""

# Standard
from typing import Any, Dict, List, Optional, Tuple
import re
import random
import string

# Third Party
from datasets import Dataset
from pydantic import Field, field_validator
import spacy

# Local
from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "PrivacyAgentBlock",
    "rag_evaluation",
    "Privacy agent that detects and masks sensitive information across multiple domains",
)
class PrivacyAgentBlock(BaseBlock):
    """Privacy Agent for sensitive information detection and masking.

    This block implements the Privacy Agent component of the multi-agent RAG evaluation
    framework. It detects and masks sensitive information including:
    1. Named entities (persons, organizations, locations)
    2. Domain-specific sensitive data (medical, financial, legal)
    3. Personal identifiers (emails, phone numbers, SSNs)
    4. Custom privacy patterns

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Input columns: ["document", "domain"]
    output_cols : List[str]
        Output columns: ["privacy_masked_document", "privacy_entities", "privacy_score"]
    privacy_domains : List[str], optional
        List of privacy-sensitive domains to detect, by default ["medical", "financial", "personal", "legal"]
    masking_strategy : str, optional
        Strategy for masking sensitive information, by default "entity_replacement"
    entity_types : List[str], optional
        Types of entities to detect and mask, by default comprehensive list
    spacy_model : str, optional
        SpaCy model for NER, by default "en_core_web_sm"
    preserve_structure : bool, optional
        Whether to preserve document structure during masking, by default True

    Examples
    --------
    >>> block = PrivacyAgentBlock(
    ...     block_name="privacy_detection_masking",
    ...     input_cols=["document", "domain"],
    ...     output_cols=["privacy_masked_document", "privacy_entities", "privacy_score"],
    ...     privacy_domains=["medical", "financial"],
    ...     masking_strategy="synthetic_replacement"
    ... )
    """

    # Core configuration
    privacy_domains: List[str] = Field(
        ["medical", "financial", "personal", "legal"],
        description="List of privacy-sensitive domains to detect"
    )
    masking_strategy: str = Field(
        "entity_replacement",
        description="Strategy for masking sensitive information"
    )
    entity_types: List[str] = Field(
        ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PHONE", "EMAIL", "SSN", "CREDIT_CARD"],
        description="Types of entities to detect and mask"
    )
    spacy_model: str = Field(
        "en_core_web_sm",
        description="SpaCy model for named entity recognition"
    )
    preserve_structure: bool = Field(
        True,
        description="Whether to preserve document structure during masking"
    )

    # Internal components (excluded from serialization)
    nlp: Optional[Any] = Field(None, exclude=True)
    privacy_patterns: Dict[str, List[str]] = Field(default_factory=dict, exclude=True)

    @field_validator("input_cols")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate input columns."""
        if v != ["document", "domain"]:
            raise BlockValidationError(
                f"PrivacyAgentBlock requires input_cols=['document', 'domain'], got {v}"
            )
        return v

    @field_validator("output_cols")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate output columns."""
        expected = ["privacy_masked_document", "privacy_entities", "privacy_score"]
        if v != expected:
            raise BlockValidationError(
                f"PrivacyAgentBlock requires output_cols={expected}, got {v}"
            )
        return v

    @field_validator("masking_strategy")
    @classmethod
    def validate_masking_strategy(cls, v):
        """Validate masking strategy."""
        supported_strategies = ["entity_replacement", "redaction", "synthetic_replacement"]
        if v not in supported_strategies:
            raise BlockValidationError(
                f"masking_strategy must be one of {supported_strategies}, got {v}"
            )
        return v

    def _initialize_components(self):
        """Initialize SpaCy NLP pipeline and privacy patterns."""
        if self.nlp is None:
            try:
                logger.info(f"Loading SpaCy model: {self.spacy_model}")
                self.nlp = spacy.load(self.spacy_model)
            except OSError:
                logger.warning(f"SpaCy model {self.spacy_model} not found. Using blank model.")
                self.nlp = spacy.blank("en")
        
        if not self.privacy_patterns:
            self._initialize_privacy_patterns()

    def _initialize_privacy_patterns(self):
        """Initialize regex patterns for detecting sensitive information."""
        self.privacy_patterns = {
            "EMAIL": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "PHONE": [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
                r'\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'
            ],
            "SSN": [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b'
            ],
            "CREDIT_CARD": [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            ],
            "MEDICAL": [
                r'\b(?:patient|diagnosis|treatment|medication|prescription|doctor|physician|hospital|clinic)\b',
                r'\b(?:blood pressure|heart rate|temperature|weight|height)\s*:?\s*\d+',
                r'\b(?:mg|ml|cc|units?)\b'
            ],
            "FINANCIAL": [
                r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
                r'\b(?:account|routing|balance|income|salary|wage)\b',
                r'\b\d{10,12}\b'  # Account numbers
            ],
            "LEGAL": [
                r'\b(?:case|docket|court|judge|attorney|lawyer|legal|lawsuit)\s+(?:number|#)?\s*\d+',
                r'\b(?:plaintiff|defendant|witness|testimony)\b'
            ]
        }

    def _detect_entities(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """Detect sensitive entities in text."""
        entities = []
        
        # SpaCy NER
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": 1.0,
                    "source": "spacy"
                })
        
        # Pattern-based detection
        for entity_type, patterns in self.privacy_patterns.items():
            if entity_type in self.entity_types:
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        entities.append({
                            "text": match.group(),
                            "label": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.8,
                            "source": "regex"
                        })
        
        # Domain-specific detection
        if domain.lower() in [d.lower() for d in self.privacy_domains]:
            domain_entities = self._detect_domain_specific_entities(text, domain.lower())
            entities.extend(domain_entities)
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x["start"])
        
        return entities

    def _detect_domain_specific_entities(self, text: str, domain: str) -> List[Dict[str, Any]]:
        """Detect domain-specific sensitive entities."""
        entities = []
        
        domain_patterns = {
            "medical": [
                r'\b(?:HIV|AIDS|cancer|diabetes|depression|anxiety)\b',
                r'\b(?:prescription|dosage|treatment|therapy)\s+\w+',
                r'\b\d+\s*(?:mg|ml|cc|units?)\b'
            ],
            "financial": [
                r'\b(?:credit score|income|debt|loan|mortgage)\s*:?\s*\$?\d+',
                r'\b(?:bank|account|routing)\s+number\s*:?\s*\d+',
                r'\b(?:investment|portfolio|assets)\s+worth\s+\$\d+'
            ],
            "legal": [
                r'\b(?:criminal|civil|family)\s+court\s+case\s+\d+',
                r'\b(?:guilty|innocent|verdict|sentence)\b',
                r'\b(?:probation|parole|custody|alimony)\b'
            ],
            "personal": [
                r'\b(?:birthday|birth date|DOB)\s*:?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
                r'\b(?:address|home|residence)\s*:?\s*\d+\s+\w+\s+(?:street|st|avenue|ave|road|rd)',
                r'\b(?:married|divorced|single|widowed)\b'
            ]
        }
        
        if domain in domain_patterns:
            for pattern in domain_patterns[domain]:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entities.append({
                        "text": match.group(),
                        "label": f"{domain.upper()}_SENSITIVE",
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.9,
                        "source": "domain_specific"
                    })
        
        return entities

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping entities, keeping the one with highest confidence."""
        if not entities:
            return entities
        
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda x: (x["start"], -x["confidence"]))
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity["start"] < existing["end"] and entity["end"] > existing["start"]):
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated

    def _mask_entities(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Mask detected entities based on the masking strategy."""
        if not entities:
            return text
        
        # Sort entities by start position in reverse order for safe replacement
        entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
        
        masked_text = text
        for entity in entities_sorted:
            start, end = entity["start"], entity["end"]
            original_text = entity["text"]
            label = entity["label"]
            
            if self.masking_strategy == "redaction":
                replacement = "[REDACTED]"
            elif self.masking_strategy == "entity_replacement":
                replacement = f"[{label}]"
            elif self.masking_strategy == "synthetic_replacement":
                replacement = self._generate_synthetic_replacement(original_text, label)
            else:
                replacement = "[MASKED]"
            
            # Preserve structure if requested
            if self.preserve_structure and len(replacement) != len(original_text):
                if len(replacement) < len(original_text):
                    replacement = replacement.ljust(len(original_text))
                else:
                    replacement = replacement[:len(original_text)]
            
            masked_text = masked_text[:start] + replacement + masked_text[end:]
        
        return masked_text

    def _generate_synthetic_replacement(self, original_text: str, label: str) -> str:
        """Generate synthetic replacement for masked entities."""
        replacements = {
            "PERSON": ["John Doe", "Jane Smith", "Alex Johnson", "Chris Brown"],
            "ORG": ["Acme Corp", "Global Inc", "Tech Solutions", "Business LLC"],
            "GPE": ["Springfield", "Riverside", "Greenville", "Madison"],
            "EMAIL": ["user@example.com", "contact@domain.org", "info@company.net"],
            "PHONE": ["555-0123", "555-0456", "555-0789"],
            "DATE": ["01/01/2020", "12/31/2021", "06/15/2022"],
            "MONEY": ["$1,000", "$5,000", "$10,000"],
        }
        
        if label in replacements:
            return random.choice(replacements[label])
        elif label.endswith("_SENSITIVE"):
            return "[SENSITIVE_INFO]"
        else:
            return f"[{label}]"

    def _compute_privacy_score(self, original_text: str, entities: List[Dict[str, Any]]) -> float:
        """Compute privacy score based on detected entities."""
        if not entities:
            return 1.0  # Perfect privacy score if no sensitive entities detected
        
        # Calculate score based on:
        # 1. Number of entities detected
        # 2. Confidence of detections
        # 3. Severity of entity types
        
        severity_weights = {
            "SSN": 1.0,
            "CREDIT_CARD": 1.0,
            "EMAIL": 0.7,
            "PHONE": 0.7,
            "PERSON": 0.5,
            "ORG": 0.3,
            "GPE": 0.2,
            "DATE": 0.2,
            "MONEY": 0.4,
            "MEDICAL_SENSITIVE": 0.9,
            "FINANCIAL_SENSITIVE": 0.8,
            "LEGAL_SENSITIVE": 0.7,
            "PERSONAL_SENSITIVE": 0.6,
        }
        
        total_severity = 0.0
        for entity in entities:
            label = entity["label"]
            confidence = entity["confidence"]
            weight = severity_weights.get(label, 0.5)
            total_severity += weight * confidence
        
        # Normalize by text length (longer texts can have more entities)
        text_length_factor = len(original_text) / 1000.0  # Normalize per 1000 characters
        normalized_severity = total_severity / max(1.0, text_length_factor)
        
        # Convert to privacy score (higher is better)
        privacy_score = max(0.0, 1.0 - min(1.0, normalized_severity))
        
        return privacy_score

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate privacy-masked documents with entity detection.

        Parameters
        ----------
        samples : Dataset
            Input dataset with 'document' and 'domain' columns.

        Returns
        -------
        Dataset
            Dataset with privacy-masked documents and privacy analysis.
        """
        # Initialize components
        self._initialize_components()
        
        # Extract data
        documents = samples["document"]
        domains = samples["domain"]
        
        logger.info(f"Processing {len(documents)} documents for privacy analysis")
        
        masked_documents = []
        all_entities = []
        privacy_scores = []
        
        for i, (document, domain) in enumerate(zip(documents, domains)):
            if i % 100 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")
            
            # Detect entities
            entities = self._detect_entities(document, domain)
            
            # Mask document
            masked_document = self._mask_entities(document, entities)
            
            # Compute privacy score
            privacy_score = self._compute_privacy_score(document, entities)
            
            masked_documents.append(masked_document)
            all_entities.append(entities)
            privacy_scores.append(privacy_score)
        
        # Log statistics
        total_entities = sum(len(entities) for entities in all_entities)
        avg_privacy_score = sum(privacy_scores) / len(privacy_scores) if privacy_scores else 0.0
        
        logger.info(f"Privacy analysis completed:")
        logger.info(f"  - Total entities detected: {total_entities}")
        logger.info(f"  - Average entities per document: {total_entities / len(documents):.2f}")
        logger.info(f"  - Average privacy score: {avg_privacy_score:.3f}")
        
        # Add results to dataset
        # Create new dataset with original data plus new columns
        new_data = {}
        for col in samples.column_names:
            new_data[col] = samples[col]
        
        new_data["privacy_masked_document"] = masked_documents
        new_data["privacy_entities"] = all_entities
        new_data["privacy_score"] = privacy_scores
        
        return Dataset.from_dict(new_data)
