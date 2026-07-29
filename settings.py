from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIG_PATH = CONFIG_DIR / "dataset.yml"
AI_MODELS_CONFIG_PATH = CONFIG_DIR / "ai_models.yml"
PROMPTS_CONFIG_PATH = CONFIG_DIR / "dataset_generator_prompt.yml"

ClassDistribution = Literal["balanced", "random"] | dict[str, float]

class DatasetSettings(BaseModel):
    input_source: str
    class_labels: list[str]
    data_col_name: str
    label_col_name: str
    class_distribution: ClassDistribution

    @model_validator(mode="after")
    def validate_class_distribution(self):
        if not isinstance(self.class_distribution, dict):
            return self

        configured_labels = set(self.class_distribution)
        expected_labels = set(self.class_labels)

        if configured_labels != expected_labels:
            missing_labels = expected_labels.difference(configured_labels)
            unknown_labels = configured_labels.difference(expected_labels)
            raise ValueError(
                "class_distribution keys must match class_labels. "
                f"Missing labels: {sorted(missing_labels)}. "
                f"Unknown labels: {sorted(unknown_labels)}."
            )

        if any(percentage < 0 for percentage in self.class_distribution.values()):
            raise ValueError("class_distribution percentages must be non-negative")

        total_percentage = sum(self.class_distribution.values())
        if abs(total_percentage - 100.0) > 1e-6:
            raise ValueError(
                "class_distribution percentages must sum to 100. "
                f"Current sum: {total_percentage}."
            )

        return self

class OutputDatasetSettings(BaseModel):
    target_size: int
    name: str


class AIModelSettings(BaseModel):
    model_name: str
    provider: str
    generation_params: dict[str, Any] = Field(default_factory=dict)


class AIModelsSettings(BaseModel):
    generation: AIModelSettings
    validation: AIModelSettings
    embedding: AIModelSettings


class PromptTemplateSettings(BaseModel):
    system: str
    user: str


class Settings(BaseModel):
    source_dataset: DatasetSettings
    target_transformation_examples_dataset: DatasetSettings
    output_dataset: OutputDatasetSettings
    ai_models: AIModelsSettings
    prompts: PromptTemplateSettings


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config() -> Settings:
    dataset_config = _read_yaml(CONFIG_PATH)
    ai_models_config = _read_yaml(AI_MODELS_CONFIG_PATH)
    prompts_config = _read_yaml(PROMPTS_CONFIG_PATH)

    return Settings(
        **dataset_config,
        ai_models=ai_models_config,
        prompts=prompts_config,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first call)."""
    return _load_config()


# Global singleton instance: `from settings import settings`
settings = get_settings()
