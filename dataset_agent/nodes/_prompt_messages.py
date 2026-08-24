from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from dataset_agent.models.dataset_generation_state_model import DatasetState


def _candidate_text(state: DatasetState) -> str:
    candidate = state.regenerated_prompt or state.generated_prompt
    if candidate is None:
        raise ValueError("A generated prompt is required for validation or repair.")
    return candidate.text


def build_messages(
    state: DatasetState,
    prompt_settings,
    validation_feedback: str | None = None,
) -> list[BaseMessage]:
    if state.source is None or state.target is None:
        raise ValueError("A source prompt and target attack class are required.")

    arguments = {
        "original_intent": state.source.original_intent,
        "original_prompt": state.source.original_prompt,
        "target_intent": state.target.target_intent,
        "target_examples": state.target.target_examples,
        "candidate_prompt": _candidate_text(state),
    }
    if validation_feedback is not None:
        arguments["validation_reason"] = validation_feedback

    return [
        SystemMessage(content=prompt_settings.system),
        HumanMessage(content=prompt_settings.user.format(**arguments)),
    ]