from dataset_agent.core.writers.writers import DatasetWriter, SyntheticRecord
from dataset_agent.models.dataset_generation_state_model import DatasetState

from settings import settings

from loguru import logger


node_logger = logger.bind(node="save")


class SaveNode:
    def __init__(self, writer: DatasetWriter):
        self.writer = writer

    def __call__(self, state: DatasetState) -> DatasetState:
        output = (
            state.regenerated_prompt
            if state.regenerated_prompt is not None
            else state.generated_prompt
        )


        source = state.source
        target = state.target

        assert source != None
        assert target != None
        assert output != None
        
        record = SyntheticRecord(
            source=source,
            target=target,
            output=output,
        )

        generated_count = state.index_to_generate
        saved_count = generated_count + 1
        last_checkpoint_index = state.last_checkpoint_index
        save_checks = settings.workflow.save_checks

        self.writer.append_at(generated_count, record)

        node_logger.success(
            "Saved record {}/{} using {} output",
            generated_count + 1,
            state.target_size,
            "repaired" if state.regenerated_prompt is not None else "generated",
        )

        last_generation = saved_count == state.target_size

        checkpoint_due = (
            last_generation
            or saved_count - last_checkpoint_index >= save_checks
        )
        
        if checkpoint_due:
            self.writer.serialize(
                start=last_checkpoint_index,
                stop=saved_count,
            )
            node_logger.info(
                "Checkpoint written for records {} through {}",
                last_checkpoint_index + 1,
                saved_count,
            )

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
                "last_checkpoint_index": saved_count if checkpoint_due else last_checkpoint_index,
            }
        )


class FlushNode:
    def __init__(self, writer: DatasetWriter):
        self.writer = writer

    def __call__(self, state: DatasetState) -> DatasetState:
        if state.last_checkpoint_index < state.index_to_generate:
            self.writer.serialize(
                start=state.last_checkpoint_index,
                stop=state.index_to_generate,
            )
            node_logger.info(
                "Final checkpoint written for records {} through {}",
                state.last_checkpoint_index + 1,
                state.index_to_generate,
            )

        return state

def save_router(state: DatasetState) -> str:
    if state.index_to_generate == state.target_size:
        return "completed"

    return "not_completed"
