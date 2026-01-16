# SPDX-License-Identifier: Apache-2.0
"""Tests for column cleanup feature."""

from sdg_hub.core.blocks.transform.duplicate_columns import DuplicateColumnsBlock
from sdg_hub.core.blocks.transform.text_concat import TextConcatBlock
from sdg_hub.core.flow.base import Flow
from sdg_hub.core.flow.column_tracker import ColumnDependencyTracker
from sdg_hub.core.flow.metadata import FlowMetadata
import pandas as pd
import pytest


class TestFlowMetadataColumnCleanup:
    """Tests for FlowMetadata column cleanup fields."""

    def test_output_columns_default_none(self):
        """Test that output_columns defaults to None."""
        metadata = FlowMetadata(name="test")
        assert metadata.output_columns is None

    def test_optimize_memory_default_false(self):
        """Test that optimize_memory defaults to False."""
        metadata = FlowMetadata(name="test")
        assert metadata.optimize_memory is False

    def test_output_columns_set(self):
        """Test setting output_columns."""
        metadata = FlowMetadata(
            name="test",
            output_columns=["col1", "col2"],
        )
        assert metadata.output_columns == ["col1", "col2"]

    def test_optimize_memory_set(self):
        """Test setting optimize_memory."""
        metadata = FlowMetadata(
            name="test",
            optimize_memory=True,
        )
        assert metadata.optimize_memory is True

    def test_output_columns_strips_whitespace(self):
        """Test that output_columns strips whitespace."""
        metadata = FlowMetadata(
            name="test",
            output_columns=["  col1  ", "col2  "],
        )
        assert metadata.output_columns == ["col1", "col2"]

    def test_output_columns_rejects_duplicates(self):
        """Test that duplicate column names are rejected."""
        with pytest.raises(ValueError, match="duplicate"):
            FlowMetadata(
                name="test",
                output_columns=["col1", "col1"],
            )

    def test_optimize_memory_without_output_columns_warns(self, caplog):
        """Test warning when optimize_memory=True without output_columns."""
        FlowMetadata(
            name="test",
            optimize_memory=True,
            output_columns=None,
        )
        assert "optimize_memory=True requires output_columns" in caplog.text


class TestColumnDependencyTracker:
    """Tests for ColumnDependencyTracker."""

    def test_build_dependency_graph_simple(self):
        """Test building dependency graph with simple blocks."""
        block1 = TextConcatBlock(
            block_name="b1", input_cols=["a", "b"], output_cols="ab"
        )
        block2 = TextConcatBlock(
            block_name="b2", input_cols=["ab", "c"], output_cols="abc"
        )

        tracker = ColumnDependencyTracker(
            blocks=[block1, block2],
            columns_to_keep={"abc"},
            original_columns={"a", "b", "c"},
        )

        assert tracker.last_consumer["a"] == 0
        assert tracker.last_consumer["b"] == 0
        assert tracker.last_consumer["ab"] == 1
        assert tracker.last_consumer["c"] == 1

    def test_extract_input_columns_string(self):
        """Test extracting input columns from string."""
        result = ColumnDependencyTracker._extract_input_columns("col1")
        assert result == ["col1"]

    def test_extract_input_columns_list(self):
        """Test extracting input columns from list."""
        result = ColumnDependencyTracker._extract_input_columns(["col1", "col2"])
        assert result == ["col1", "col2"]

    def test_extract_input_columns_dict(self):
        """Test extracting input columns from dict (keys are source cols)."""
        result = ColumnDependencyTracker._extract_input_columns({"src": "dst"})
        assert result == ["src"]

    def test_extract_input_columns_none(self):
        """Test extracting input columns from None."""
        result = ColumnDependencyTracker._extract_input_columns(None)
        assert result == []

    def test_get_droppable_columns_preserves_final(self):
        """Test that final output columns are never dropped."""
        block1 = TextConcatBlock(block_name="b1", input_cols=["a"], output_cols="final")

        tracker = ColumnDependencyTracker(
            blocks=[block1],
            columns_to_keep={"final"},
            original_columns={"a"},
        )

        droppable = tracker.get_droppable_columns(0, {"a", "final"})
        assert "final" not in droppable
        assert "a" not in droppable  # original preserved

    def test_get_droppable_columns_preserves_original(self):
        """Test that original columns are never dropped."""
        block1 = TextConcatBlock(block_name="b1", input_cols=["a"], output_cols="temp")

        tracker = ColumnDependencyTracker(
            blocks=[block1],
            columns_to_keep={"output"},
            original_columns={"a"},
        )

        droppable = tracker.get_droppable_columns(0, {"a", "temp"})
        assert "a" not in droppable
        assert "temp" in droppable

    def test_get_droppable_columns_after_last_consumer(self):
        """Test dropping columns after their last consumer."""
        block1 = TextConcatBlock(
            block_name="b1", input_cols=["a", "b"], output_cols="ab"
        )
        block2 = TextConcatBlock(
            block_name="b2", input_cols=["ab"], output_cols="final"
        )

        tracker = ColumnDependencyTracker(
            blocks=[block1, block2],
            columns_to_keep={"final"},
            original_columns={"a", "b"},
        )

        # After block 0: 'ab' still needed by block 1
        droppable0 = tracker.get_droppable_columns(0, {"a", "b", "ab"})
        assert "ab" not in droppable0

        # After block 1: 'ab' no longer needed
        droppable1 = tracker.get_droppable_columns(1, {"a", "b", "ab", "final"})
        assert "ab" in droppable1

    def test_get_droppable_columns_never_consumed(self):
        """Test that columns never consumed are droppable."""
        block1 = TextConcatBlock(
            block_name="b1", input_cols=["a"], output_cols="unused"
        )
        block2 = TextConcatBlock(block_name="b2", input_cols=["a"], output_cols="final")

        tracker = ColumnDependencyTracker(
            blocks=[block1, block2],
            columns_to_keep={"final"},
            original_columns={"a"},
        )

        # 'unused' is never consumed, can be dropped after creation
        droppable = tracker.get_droppable_columns(0, {"a", "unused"})
        assert "unused" in droppable


class TestFlowColumnCleanup:
    """Tests for Flow column cleanup during execution."""

    def test_flow_without_output_columns_keeps_all(self):
        """Test that without output_columns, all columns are kept."""
        flow = Flow(
            metadata=FlowMetadata(name="test"),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="ab",
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        result = flow.generate(dataset)

        assert "a" in result.columns
        assert "b" in result.columns
        assert "ab" in result.columns

    def test_flow_with_output_columns_drops_intermediate(self):
        """Test that intermediate columns are dropped."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["final"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="step1",
                    input_cols=["a", "b"],
                    output_cols="intermediate",
                ),
                DuplicateColumnsBlock(
                    block_name="step2",
                    input_cols={"intermediate": "final"},
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        result = flow.generate(dataset)

        # Original columns preserved
        assert "a" in result.columns
        assert "b" in result.columns
        # Final output kept
        assert "final" in result.columns
        # Intermediate dropped
        assert "intermediate" not in result.columns

    def test_flow_with_optimize_memory_drops_early(self):
        """Test that optimize_memory drops columns during execution."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["final"],
                optimize_memory=True,
            ),
            blocks=[
                TextConcatBlock(
                    block_name="step1",
                    input_cols=["a", "b"],
                    output_cols="temp1",
                ),
                TextConcatBlock(
                    block_name="step2",
                    input_cols=["temp1", "c"],
                    output_cols="temp2",
                ),
                DuplicateColumnsBlock(
                    block_name="step3",
                    input_cols={"temp2": "final"},
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"], "c": ["3"]})
        result = flow.generate(dataset)

        # Original columns preserved
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        # Final output kept
        assert "final" in result.columns
        # All intermediates dropped
        assert "temp1" not in result.columns
        assert "temp2" not in result.columns

    def test_flow_missing_output_columns_warns(self, caplog):
        """Test warning when output_columns specifies missing columns."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["nonexistent", "output"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="output",
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        flow.generate(dataset)

        assert "columns_to_keep not found in dataset" in caplog.text
        assert "nonexistent" in caplog.text

    def test_flow_optimize_memory_without_output_columns_no_effect(self):
        """Test that optimize_memory without output_columns has no effect."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                optimize_memory=True,
                output_columns=None,
            ),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="ab",
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        result = flow.generate(dataset)

        # All columns kept since output_columns is None
        assert "a" in result.columns
        assert "b" in result.columns
        assert "ab" in result.columns


class TestFlowAllOutputColumns:
    """Tests for Flow.all_output_columns property."""

    def test_all_output_columns_empty_flow(self):
        """Test all_output_columns with no blocks."""
        flow = Flow(
            metadata=FlowMetadata(name="test"),
            blocks=[],
        )
        assert flow.all_output_columns == set()

    def test_all_output_columns_single_block(self):
        """Test all_output_columns with a single block."""
        flow = Flow(
            metadata=FlowMetadata(name="test"),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="ab",
                ),
            ],
        )
        assert flow.all_output_columns == {"ab"}

    def test_all_output_columns_multiple_blocks(self):
        """Test all_output_columns with multiple blocks."""
        flow = Flow(
            metadata=FlowMetadata(name="test"),
            blocks=[
                TextConcatBlock(
                    block_name="step1",
                    input_cols=["a", "b"],
                    output_cols="ab",
                ),
                DuplicateColumnsBlock(
                    block_name="step2",
                    input_cols={"ab": "final"},
                ),
            ],
        )
        assert flow.all_output_columns == {"ab", "final"}

    def test_all_output_columns_list_output(self):
        """Test all_output_columns with list output_cols."""
        flow = Flow(
            metadata=FlowMetadata(name="test"),
            blocks=[
                DuplicateColumnsBlock(
                    block_name="dup",
                    input_cols={"a": "b", "c": "d"},
                ),
            ],
        )
        assert flow.all_output_columns == {"b", "d"}


class TestFlowColumnsToKeep:
    """Tests for columns_to_keep parameter in flow.generate()."""

    def test_columns_to_keep_superset_of_output_columns(self):
        """Test that columns_to_keep can include more than output_columns."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["final"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="step1",
                    input_cols=["a", "b"],
                    output_cols="intermediate",
                ),
                DuplicateColumnsBlock(
                    block_name="step2",
                    input_cols={"intermediate": "final"},
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        # Request more columns than minimal
        result = flow.generate(dataset, columns_to_keep=["final", "intermediate"])

        assert "final" in result.columns
        assert "intermediate" in result.columns
        assert "a" in result.columns  # original preserved
        assert "b" in result.columns  # original preserved

    def test_columns_to_keep_missing_output_columns_raises(self):
        """Test that columns_to_keep must include all output_columns."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["col1", "col2"],
            ),
            blocks=[
                DuplicateColumnsBlock(
                    block_name="dup",
                    input_cols={"a": "col1", "b": "col2"},
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        with pytest.raises(ValueError, match="must include all output_columns"):
            flow.generate(dataset, columns_to_keep=["col1"])  # missing col2

    def test_columns_to_keep_with_unknown_columns_raises(self):
        """Test error when columns_to_keep has unknown columns."""
        flow = Flow(
            metadata=FlowMetadata(
                name="test",
                output_columns=["output"],
            ),
            blocks=[
                TextConcatBlock(
                    block_name="concat",
                    input_cols=["a", "b"],
                    output_cols="output",
                ),
            ],
        )

        dataset = pd.DataFrame({"a": ["1"], "b": ["2"]})
        with pytest.raises(ValueError, match="columns not produced by flow"):
            flow.generate(dataset, columns_to_keep=["output", "nonexistent"])


class TestColumnDependencyTrackerEdgeCases:
    """Edge case tests for ColumnDependencyTracker."""

    def test_empty_blocks_list(self):
        """Test tracker with empty blocks list."""
        tracker = ColumnDependencyTracker(
            blocks=[],
            columns_to_keep={"output"},
            original_columns={"input"},
        )
        assert tracker.last_consumer == {}
        droppable = tracker.get_droppable_columns(0, {"input", "output", "temp"})
        assert "temp" in droppable
        assert "input" not in droppable
        assert "output" not in droppable
