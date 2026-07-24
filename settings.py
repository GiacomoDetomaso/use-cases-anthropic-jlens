from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "dataset.yml"


class SourceDatasetSettings(BaseModel):
    input_source: str
    class_labels: list[str]
    data_col_name: str
    label_col_name: str
    class_distribution: str


class TargetTransformationExamplesDatasetSettings(BaseModel):
    input_source: str
    class_labels: list[str]
    data_col_name: str
    label_col_name: str


class OutputDatasetSettings(BaseModel):
    target_size: int
    name: str


class Settings(BaseModel):
    source_dataset: SourceDatasetSettings
    target_transformation_examples_dataset: TargetTransformationExamplesDatasetSettings
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
