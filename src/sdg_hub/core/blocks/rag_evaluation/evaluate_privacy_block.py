# SPDX-License-Identifier: Apache-2.0
"""Privacy Evaluation Block for assessing privacy compliance in generated QA pairs.

This module implements privacy evaluation for the multi-agent RAG evaluation framework,
assessing whether generated questions and answers maintain privacy compliance.
"""

# Standard
from typing import Any, List, Optional, Union

# Third Party
from datasets import Dataset
from pydantic import ConfigDict, Field, field_validator

# Local
from ...utils.error_handling import BlockValidationError
from ...utils.logger_config import setup_logger
from ..base import BaseBlock
from ..filtering.column_value_filter import ColumnValueFilterBlock
from ..llm.llm_chat_block import LLMChatBlock
from ..llm.prompt_builder_block import PromptBuilderBlock
from ..llm.text_parser_block import TextParserBlock
from ..registry import BlockRegistry

logger = setup_logger(__name__)


@BlockRegistry.register(
    "EvaluatePrivacyBlock",
    "rag_evaluation",
    "Thin wrapper composing 4 blocks for privacy compliance evaluation of generated QA pairs",
)
class EvaluatePrivacyBlock(BaseBlock):
    """Thin wrapper for privacy evaluation using composed blocks.

    Composes PromptBuilderBlock + LLMChatBlock + TextParserBlock + ColumnValueFilterBlock
    into a single privacy evaluation pipeline with smart parameter routing.

    Parameters
    ----------
    block_name : str
        Name of the block.
    input_cols : List[str]
        Input columns: ["question", "answer", "privacy_entities"]
    output_cols : List[str]
        Output columns: ["privacy_explanation", "privacy_compliance"]
    model : Optional[str]
        LLM model identifier.
    api_base : Optional[str]
        API base URL.
    api_key : Optional[str]
        API key.
    prompt_config_path : str
        Path to YAML prompt template file (required).
    **kwargs : Any
        All other parameters are automatically routed to appropriate internal blocks
        based on each block's accepted parameters.
    """

    model_config = ConfigDict(
        extra="allow"
    )  # Allow extra fields for dynamic forwarding

    # --- Core configuration ---
    prompt_config_path: str = Field(
        ...,
        description="Path to YAML file containing the privacy evaluation prompt template",
    )

    # --- LLM interface (for flow detection) ---
    model: Optional[str] = Field(None, description="LLM model identifier")
    api_base: Optional[str] = Field(None, description="API base URL")
    api_key: Optional[str] = Field(None, description="API key")

    # --- Filter configuration ---
    filter_value: Union[str, int, float] = Field(
        "COMPLIANT", description="Value to filter on for privacy compliance"
    )
    operation: str = Field("eq", description="Filter operation")
    convert_dtype: Optional[str] = Field(
        None, description="Data type conversion for filter column"
    )

    # --- Parser configuration ---
    start_tags: list[str] = Field(
        ["[Start of Analysis]", "[Start of Compliance]"],
        description="Start tags for parsing analysis and compliance status",
    )
    end_tags: list[str] = Field(
        ["[End of Analysis]", "[End of Compliance]"],
        description="End tags for parsing analysis and compliance status",
    )
    parsing_pattern: Optional[str] = Field(
        None,
        description="Regex pattern for custom parsing. If provided, takes precedence over tag-based parsing",
    )

    # --- Internal blocks (composition) ---
    prompt_builder: PromptBuilderBlock = Field(None, exclude=True)  # type: ignore
    llm_chat: LLMChatBlock = Field(None, exclude=True)  # type: ignore
    text_parser: TextParserBlock = Field(None, exclude=True)  # type: ignore
    filter_block: ColumnValueFilterBlock = Field(None, exclude=True)  # type: ignore

    @field_validator("input_cols")
    @classmethod
    def validate_input_cols(cls, v):
        """Validate input columns."""
        if v != ["question", "answer", "privacy_entities"]:
            raise BlockValidationError(
                f"EvaluatePrivacyBlock requires input_cols=['question', 'answer', 'privacy_entities'], got {v}"
            )
        return v

    @field_validator("output_cols")
    @classmethod
    def validate_output_cols(cls, v):
        """Validate output columns."""
        if v != ["privacy_explanation", "privacy_compliance"]:
            raise BlockValidationError(
                f"EvaluatePrivacyBlock requires output_cols=['privacy_explanation', 'privacy_compliance'], got {v}"
            )
        return v

    def _initialize_blocks(self):
        """Initialize internal blocks with parameter routing."""
        # Get all model fields for parameter routing
        all_params = self.model_dump()
        
        # Remove our own fields
        internal_params = {k: v for k, v in all_params.items() 
                          if k not in {"prompt_config_path", "filter_value", "operation", 
                                     "convert_dtype", "start_tags", "end_tags", "parsing_pattern"}}
        
        # Initialize PromptBuilderBlock
        self.prompt_builder = PromptBuilderBlock(
            block_name=f"{self.block_name}_prompt_builder",
            input_cols=["question", "answer", "privacy_entities"],
            output_cols="privacy_evaluation_prompt",
            prompt_config_path=self.prompt_config_path,
        )
        
        # Initialize LLMChatBlock with routed parameters
        llm_params = {k: v for k, v in internal_params.items() 
                     if k in LLMChatBlock.model_fields}
        self.llm_chat = LLMChatBlock(
            block_name=f"{self.block_name}_llm_chat",
            input_cols="privacy_evaluation_prompt",
            output_cols="raw_privacy_evaluation",
            **llm_params
        )
        
        # Initialize TextParserBlock
        parser_params = {
            "start_tags": self.start_tags,
            "end_tags": self.end_tags,
        }
        if self.parsing_pattern:
            parser_params["parsing_pattern"] = self.parsing_pattern
        
        self.text_parser = TextParserBlock(
            block_name=f"{self.block_name}_text_parser",
            input_cols="raw_privacy_evaluation",
            output_cols=["privacy_explanation", "privacy_compliance"],
            **parser_params
        )
        
        # Initialize ColumnValueFilterBlock
        self.filter_block = ColumnValueFilterBlock(
            block_name=f"{self.block_name}_filter",
            input_cols="privacy_compliance",
            filter_column="privacy_compliance",
            filter_value=self.filter_value,
            operation=self.operation,
            convert_dtype=self.convert_dtype,
        )

    def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
        """Generate privacy evaluation for QA pairs.

        Parameters
        ----------
        samples : Dataset
            Input dataset with question, answer, and privacy_entities columns.

        Returns
        -------
        Dataset
            Dataset with privacy evaluation results.
        """
        # Initialize blocks if not already done
        if self.prompt_builder is None:
            self._initialize_blocks()
        
        logger.info(f"Starting privacy evaluation for {len(samples)} samples")
        
        # Step 1: Build prompts
        samples = self.prompt_builder(samples, **kwargs)
        
        # Step 2: Generate LLM responses
        samples = self.llm_chat(samples, **kwargs)
        
        # Step 3: Parse responses
        samples = self.text_parser(samples, **kwargs)
        
        # Step 4: Filter by privacy compliance
        samples = self.filter_block(samples, **kwargs)
        
        logger.info(f"Privacy evaluation completed. {len(samples)} samples passed filtering.")
        
        return samples
