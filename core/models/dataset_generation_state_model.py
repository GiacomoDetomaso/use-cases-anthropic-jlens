from pydantic import BaseModel, Field

from core.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
    OutputModel,
)


class DocumentState(BaseModel):
    target_size: int = Field(description="The size of the dataset to generate")
    generated_count: int = Field(description="The number of the actual generated data samples. It can be lower than target size if some generation fails.")
    remaining_input_indices: list[int] = Field(default_factory=list, description="Input indices of the source dataset not utilized during generation. Useful when target_size < dataset size")

    input_distribution: dict[str, int] = Field(default_factory=dict, description="Distribution of the input classes")
    target_distribution: dict[str, int] = Field(default_factory=dict, description="Distribution of the output classes")

    source: InputDatasetModel | None = Field(default=None, description="Current input sent to the generation node")
    target: InputAttackModel = Field(description="Current target output into which transform the current_input")

    generated_prompt: OutputModel | None = Field(default=None, description="Generated prompt")

    retries: int = Field(default=0, ge=0, description="Number of retries by the decision check agent")
    # TODO check other fields
