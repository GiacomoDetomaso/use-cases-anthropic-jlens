from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, Field, computed_field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "dataset.yml"
AI_MODELS_CONFIG_PATH = CONFIG_DIR / "ai_models.yml"
PROMPTS_CONFIG_PATH = CONFIG_DIR / "dataset_generator_prompt.yml"
VALIDATOR_PROMPTS_CONFIG_PATH = CONFIG_DIR / "prompt_validator.yml"
REPAIR_PROMPTS_CONFIG_PATH = CONFIG_DIR / "prompt_repair.yml"
WORKFLOW_CONFIG_PATH = CONFIG_DIR / "workflow.yml"

ClassDistribution = Literal["balanced", "random"] | dict[str, float]


class DatasetSettings(BaseModel):
    input_source: str
    class_labels: list[str]
    data_col_name: str
    label_col_name: str
    class_distribution: ClassDistribution
    class_descriptions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_class_distribution(self) -> Self:
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
                f"class_distribution percentages must sum to 100. Current sum: {total_percentage}."
            )

        return self

    def get_class_description(self, class_name: str) -> str:
        return self.class_descriptions.get(class_name, "no description for this class")


class OutputDatasetSettings(BaseModel):
    target_size: int
    name: str
    format: Literal["csv", "jsonl"] = "csv"
    merge: Literal["all", "not_selected_indeces"] = "all"


class AIModelSettings(BaseModel):
    model_name: str
    provider: str
    generation_params: dict[str, Any] = Field(default_factory=dict)


class AIModelsSettings(BaseModel):
    generation: AIModelSettings


class PromptTemplateSettings(BaseModel):
    system: str
    user: str


class VllmParamSettings(BaseModel):
    name: str
    value: bool | str | int | float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value_str(self) -> str:
        return str(self.value)


class VllmSettings(BaseModel):
    health_sleep_seconds: float = Field(default=0.5, gt=0)
    invoke_mode: Literal["sync", "async"]
    warmup_step: bool
    vllm_params: list[VllmParamSettings]

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        names = [p.name for p in self.vllm_params]

        if len(names) != len(set(names)):
            raise ValueError("Some parameter names are duplicated. Cannot build CLI command.")

        if "port" not in names:
            raise ValueError("Mandatory `port` parameter not configured")

        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def port(self) -> int:
        for param in self.vllm_params:
            if param.name == "port":
                return int(param.value)

        raise RuntimeError("Port should always exist after validation")

    def get_vllm_cmd(self, model_name: str) -> list[str]:
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(model_name),
        ]

        for param in self.vllm_params:
            # Check strictly for boolean types
            if isinstance(param.value, bool):
                if param.value:
                    cmd.append(f"--{param.name}")
                else:
                    cmd.append(f"--no-{param.name}")
            else:
                cmd.extend([f"--{param.name}", param.value_str])

        return cmd

    def get_base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"

    def get_health_url(self) -> str:
        # The __port attribute is certainly configured since if not found a ValueError is raised
        return f"http://localhost:{self.port}/health"


class WorkflowInferenceSettings(BaseModel):
    mode: Literal["no_inference_engine", "vllm"]
    vllm: VllmSettings | None


class WorkflowSettings(BaseModel):
    generation_schema_fix_retries: int = Field(default=0, gt=0, lt=5)
    max_repair_attempts: int = Field(default=0, ge=0, lt=5)
    workers: int = Field(default=1, ge=1)
    worker_class_groups: list[list[str]] | None = None
    save_checks: int = Field(default=1, ge=1)
    resume: bool = True
    inference: WorkflowInferenceSettings


class Settings(BaseModel):
    source_dataset: DatasetSettings
    target_transformation_examples_dataset: DatasetSettings
    output_dataset: OutputDatasetSettings
    ai_models: AIModelsSettings
    prompts: PromptTemplateSettings
    validator_prompts: PromptTemplateSettings
    repair_prompts: PromptTemplateSettings
    workflow: WorkflowSettings

    @model_validator(mode="after")
    def validate_settings(self) -> Self:
        if self.workflow.save_checks > self.output_dataset.target_size:
            raise ValueError("`save_checks` can't be higher than the actual dataset_size")

        workflow = self.workflow
        if workflow.workers == 1:
            if workflow.worker_class_groups is not None:
                raise ValueError(
                    "`worker_class_groups` is only supported when `workers` is greater than 1"
                )
            return self

        if isinstance(self.source_dataset.class_distribution, dict):
            raise ValueError(
                "multi-worker generation requires source_dataset.class_distribution "
                "to be `balanced` or `random`"
            )

        if workflow.inference.mode != "vllm" or workflow.inference.vllm is None:
            raise ValueError("multi-worker generation requires a configured vLLM inference server")

        groups = workflow.worker_class_groups
        if groups is None:
            raise ValueError("multi-worker generation requires `worker_class_groups`")

        if len(groups) != workflow.workers:
            raise ValueError("`worker_class_groups` must contain exactly one group per worker")

        if any(not group for group in groups):
            raise ValueError("every worker class group must contain at least one source class")

        assigned_labels = [label for group in groups for label in group]
        if len(assigned_labels) != len(set(assigned_labels)):
            raise ValueError("source classes cannot be assigned to more than one worker")

        configured_labels = set(self.source_dataset.class_labels)
        assigned_labels_set = set(assigned_labels)
        if assigned_labels_set != configured_labels:
            missing_labels = configured_labels.difference(assigned_labels_set)
            unknown_labels = assigned_labels_set.difference(configured_labels)
            raise ValueError(
                "worker class groups must assign every source class exactly once. "
                f"Missing labels: {sorted(missing_labels)}. "
                f"Unknown labels: {sorted(unknown_labels)}."
            )

        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f)

    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")

    return cast("dict[str, Any]", loaded)


def _load_config() -> Settings:
    dataset_config = _read_yaml(CONFIG_PATH)
    ai_models_config = _read_yaml(AI_MODELS_CONFIG_PATH)
    prompts_config = _read_yaml(PROMPTS_CONFIG_PATH)
    validator_prompts_config = _read_yaml(VALIDATOR_PROMPTS_CONFIG_PATH)
    repair_prompts_config = _read_yaml(REPAIR_PROMPTS_CONFIG_PATH)
    workflow_config = _read_yaml(WORKFLOW_CONFIG_PATH)

    return Settings(
        **dataset_config,
        ai_models=AIModelsSettings.model_validate(ai_models_config),
        prompts=PromptTemplateSettings.model_validate(prompts_config),
        validator_prompts=PromptTemplateSettings.model_validate(validator_prompts_config),
        repair_prompts=PromptTemplateSettings.model_validate(repair_prompts_config),
        workflow=WorkflowSettings.model_validate(workflow_config),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton (cached after first call)."""
    return _load_config()


# Global singleton instance: `from settings import settings`
settings = get_settings()
