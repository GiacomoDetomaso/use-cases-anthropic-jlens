from pydantic import BaseModel, Field

from core.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
    OutputModel,
    QualityAssessmentModel,
)


class DistributionBucket(BaseModel):
    target: int = Field(ge=0, description="Target number of samples for the class")
    actual: int = Field(default=0, ge=0, description="Actual number of samples picked for the class")

DistributionState = dict[str, DistributionBucket]

class DatasetState(BaseModel):
    target_size: int = Field(description="The size of the dataset to generate")
    generated_count: int = Field(description="The number of the actual generated data samples. It can be lower than target size if some generation fails.")
    remaining_input_indices: list[int] = Field(default_factory=list, description="Input indices of the source dataset not utilized during generation. Useful when target_size < dataset size")

    input_distribution: DistributionState = Field(default_factory=dict, description="Distribution of the input classes")
    target_distribution: DistributionState = Field(default_factory=dict, description="Distribution of the output classes")

    source: InputDatasetModel | None = Field(default=None, description="Current input sent to the generation node")
    target: InputAttackModel | None = Field(default=None, description="Current target output into which transform the current_input")

    generated_prompt: OutputModel | None = Field(default=None, description="Generated prompt")
    regenerated_prompt: OutputModel | None = Field(default=None, description="Prompt produced by the Repair Agent when repairing a rejected generation, if any")

    should_retry: bool = Field(default=False, description="Set by the Similarity Validator or Quality Checker when the current prompt fails validation and must go through the Repair Agent")
    validation_output: QualityAssessmentModel | None = Field(default=None, description="Overall quality assessment produced by the Quality Checker for the current generated prompt")

    retries: int = Field(default=0, ge=0, description="Number of times the Repair Agent has regenerated/repaired the current record")
