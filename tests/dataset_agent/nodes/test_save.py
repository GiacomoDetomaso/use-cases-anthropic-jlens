from unittest.mock import Mock

import pytest

from dataset_agent.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
    OutputModel,
)
from dataset_agent.models.dataset_generation_state_model import DatasetState
from dataset_agent.nodes.save import SaveNode
from settings import settings


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
            OutputModel(text=regenerated_text)
            if regenerated_text is not None
            else None
        ),
    )

    SaveNode(writer)(state)

    assert writer.append.call_args.args[0].output == expected_output