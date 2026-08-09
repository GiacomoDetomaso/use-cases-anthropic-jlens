from dataset_agent.core.writers import DatasetWriter, SyntheticRecord
from dataset_agent.models.dataset_generation_state_model import DatasetState

from settings import settings

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

        generated_count = state.index_to_generate
        last_checkpoint_index = state.last_checkpoint_index
        save_checks = settings.workflow.save_checks

        self.writer.append(record)

        logger.info(f"Data point {generated_count} appended to the dataset")

        serializable = False
        last_generation = state.index_to_generate == state.target_size

        serializable = (
            last_generation
            or generated_count == save_checks
            or generated_count - last_checkpoint_index == save_checks
        )
        
        if serializable:
            if last_generation:
                start = last_checkpoint_index
            else:
                start = max(0, generated_count - save_checks)

            self.writer.serialize(
                start=start, 
                stop=generated_count
            )

            logger.info(f"Data points from {generated_count - save_checks} up to {generated_count} serialized")

        return state.model_copy(
            update={
                "index_to_generate": generated_count + 1,
                "source": None,
                "target": None,
                "generated_prompt": None,
                "regenerated_prompt": None,
                "validation_output": None,
                "should_retry": False,
                "retries": 0,
                "last_checkpoint_index": generated_count if serializable else last_checkpoint_index
            }
        )

def save_router(state: DatasetState) -> str:
    if state.index_to_generate == (state.target_size + 1):
        return "completed"

    return "not_completed"
