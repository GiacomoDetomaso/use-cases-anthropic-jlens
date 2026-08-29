from unittest.mock import Mock

import pytest

from app.settings import settings
from dataset_agent.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
    OutputModel,
)
from dataset_agent.models.dataset_generation_state_model import (
    DatasetState,
)
from dataset_agent.nodes.save import FlushNode, SaveNode


@pytest.mark.parametrize(
    ("generated_text", "regenerated_text", "expected_output"),
    [
        ("initial prompt", None, "initial prompt"),
        ("initial prompt", "repaired prompt", "repaired prompt"),
    ],
)
def test_save_uses_available_generated_output(
    generated_text: str,
    regenerated_text: str | None,
    expected_output: str,
) -> None:
    writer = Mock()
    state = DatasetState(
        target_size=2,
        index_to_generate=0,
        source=InputDatasetModel(
            original_prompt="Where is my order?",
            original_intent="get_order_status",
        ),
        target=InputAttackModel(
            target_intent=settings.target_transformation_examples_dataset.class_labels[0],
            target_examples="Example target",
        ),
        generated_prompt=OutputModel(text=generated_text),
        regenerated_prompt=(
            OutputModel(text=regenerated_text) if regenerated_text is not None else None
        ),
    )

    SaveNode(writer)(state)

    assert writer.append_at.call_args.args[1].output.text == expected_output


def test_save_serializes_the_final_record_without_duplicate_bounds() -> None:
    writer = Mock()
    state = DatasetState(
        target_size=2,
        index_to_generate=1,
        last_checkpoint_index=1,
        source=InputDatasetModel(
            original_prompt="Where is my order?",
            original_intent="get_order_status",
        ),
        target=InputAttackModel(
            target_intent=settings.target_transformation_examples_dataset.class_labels[0],
            target_examples="Example target",
        ),
        generated_prompt=OutputModel(text="final prompt"),
    )

    saved_state = SaveNode(writer)(state)

    writer.serialize.assert_called_once_with(start=1, stop=2)
    assert saved_state.last_checkpoint_index == 2


def test_save_defers_serialization_until_the_configured_interval() -> None:
    writer = Mock()
    state = DatasetState(
        target_size=settings.workflow.save_checks + 2,
        index_to_generate=1,
        last_checkpoint_index=0,
        source=InputDatasetModel(
            original_prompt="Where is my order?",
            original_intent="get_order_status",
        ),
        target=InputAttackModel(
            target_intent=settings.target_transformation_examples_dataset.class_labels[0],
            target_examples="Example target",
        ),
        generated_prompt=OutputModel(text="pending prompt"),
    )

    saved_state = SaveNode(writer)(state)

    writer.serialize.assert_not_called()
    assert saved_state.last_checkpoint_index == 0


def test_flush_serializes_records_pending_after_a_final_discard() -> None:
    writer = Mock()
    state = DatasetState(
        target_size=23,
        index_to_generate=23,
        last_checkpoint_index=0,
    )

    FlushNode(writer)(state)

    writer.serialize.assert_called_once_with(start=0, stop=23)
