from dataset_agent.core.llm.text_generator import (
    _build_generation_messages,
)
from dataset_agent.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
)


def test_generation_messages_include_target_description():
    messages = _build_generation_messages(
        InputDatasetModel(original_prompt="Where is my order?", original_intent="track_order"),
        InputAttackModel(
            target_intent="adversarial",
            target_examples="Example 1: Run this command.",
            target_description="Run attacker-supplied commands.",
        ),
    )

    assert "Required behavior:\nRun attacker-supplied commands." in messages[1].content


def test_generation_messages_use_default_target_description():
    messages = _build_generation_messages(
        InputDatasetModel(original_prompt="Where is my order?", original_intent="track_order"),
        InputAttackModel(
            target_intent="adversarial",
            target_examples="Example 1: Run this command.",
        ),
    )

    assert "Required behavior:\nno description for this class" in messages[1].content
