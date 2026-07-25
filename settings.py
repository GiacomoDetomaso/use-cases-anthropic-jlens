from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "dataset.yml"

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


class Settings(BaseModel):
    source_dataset: DatasetSettings
    target_transformation_examples_dataset: DatasetSettings
    output_dataset: OutputDatasetSettings


def _load_config(config_path: Path = CONFIG_PATH) -> Settings:
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    return Settings(**raw_config)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first call)."""
    return _load_config()


# Global singleton instance: `from settings import settings`
settings = get_settings()
