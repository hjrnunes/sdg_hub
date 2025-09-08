# SPDX-License-Identifier: Apache-2.0
"""RAG evaluation blocks for multi-agent synthetic dataset generation.

This package provides specialized blocks for implementing the multi-agent framework
described in "Diverse And Private Synthetic Datasets Generation for RAG evaluation"
by Driouich et al. (2024).
"""

from .diversity_agent_block import DiversityAgentBlock
from .privacy_agent_block import PrivacyAgentBlock
from .evaluate_diversity_block import EvaluateDiversityBlock
from .evaluate_privacy_block import EvaluatePrivacyBlock

__all__ = [
    "DiversityAgentBlock",
    "PrivacyAgentBlock", 
    "EvaluateDiversityBlock",
    "EvaluatePrivacyBlock",
]
