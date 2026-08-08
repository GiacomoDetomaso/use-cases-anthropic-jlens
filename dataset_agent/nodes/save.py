
from dataset_agent.core.writers import DatasetWriter, SyntheticRecord
from dataset_agent.models.dataset_generation_state_model import DatasetState

from loguru import logger

class SaveNode:
    def __init__(self, writer: DatasetWriter):
        self.writer = writer

    def __call__(self, state: DatasetState) -> DatasetState:
        record = SyntheticRecord(
            source=state.source,
            target=state.target,
            output=(
                state.regenerated_prompt
                or state.generated_prompt
            ),
        )

        self.writer.append(record)

        generated_count = state.generated_count

        logger.info(f"Data point {generated_count} generated")

        return state.model_copy(
            update={
                "generated_count": generated_count + 1,
                "source": None,
                "target": None,
                "generated_prompt": None,
                "regenerated_prompt": None,
                "validation_output": None,
                "should_retry": False,
                "retries": 0,
            }
        )

def save_router(state: DatasetState) -> str:
    if state.generated_count == state.target_size:
        return "completed"

    return "not_completed"
