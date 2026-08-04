from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator, computed_field

CONFIG_DIR = Path(__file__).resolve().parent / "config"
CONFIG_PATH = CONFIG_DIR / "dataset.yml"
AI_MODELS_CONFIG_PATH = CONFIG_DIR / "ai_models.yml"
PROMPTS_CONFIG_PATH = CONFIG_DIR / "dataset_generator_prompt.yml"
WORKFLOW_CONFIG_PATH = CONFIG_DIR / "workflow.yml"

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


class VllmParamSettings(BaseModel):
    name: str
    value: str | int | float

    @computed_field
    @property
    def value_str(self) -> str:
        return str(self.value)
    

class VllmSettings(BaseModel):
    health_sleep_seconds: int = Field(default=0.5, gt=0)
    invoke_mode: Literal["sync", "async"]
    vllm_params: list[VllmParamSettings]

    @model_validator(mode="after")
    def validate(self):
        names = [p.name for p in self.vllm_params]

        if len(names) != len(set(names)):
            raise ValueError(
                "Some parameter names are duplicated. Cannot build CLI command."
            )

        if "port" not in names:
            raise ValueError("Mandatory `port` parameter not configured")

        return self

    @computed_field
    @property
    def _port(self) -> int:
        for param in self.vllm_params:
            if param.name == "port":
                return int(param.value)
            
        raise RuntimeError("Port should always exist after validation")

    def get_vllm_cmd(self, model_name: str) -> list[str]:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
        ]

        for param in self.vllm_params:
            extension = [f"--{param.name}"]

            if not isinstance(param.value, bool):
                extension.append(param.value_str)

            cmd.extend(extension)

        return cmd

    def get_base_url(self):
        return f"http://localhost:{self._port}/v1"

    def get_health_url(self):
        # The __port attribute is certainly configured since if not found a ValueError is raised
        return f"http://localhost:{self._port}/health"


class WorkflowInferenceSettings(BaseModel):
    mode: Literal["no_inference_engine", "vllm"]
    vllm: VllmSettings | None


class WorkflowSettings(BaseModel):
    generation_schema_fix_retries: int = Field(default=0, gt=0, lt=5)
    workers: int = Field(default=1, ge=1)
    inference: WorkflowInferenceSettings


class Settings(BaseModel):
    source_dataset: DatasetSettings
    target_transformation_examples_dataset: DatasetSettings
    output_dataset: OutputDatasetSettings
    ai_models: AIModelsSettings
    prompts: PromptTemplateSettings
    workflow: WorkflowSettings


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config() -> Settings:
    dataset_config = _read_yaml(CONFIG_PATH)
    ai_models_config = _read_yaml(AI_MODELS_CONFIG_PATH)
    prompts_config = _read_yaml(PROMPTS_CONFIG_PATH)
    workflow_config = _read_yaml(WORKFLOW_CONFIG_PATH)

    return Settings(
        **dataset_config,
        ai_models=ai_models_config,
        prompts=prompts_config,
        workflow=workflow_config
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first call)."""
    return _load_config()


# Global singleton instance: `from settings import settings`
settings = get_settings()
