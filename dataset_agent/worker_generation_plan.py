"""Create isolated, quota-preserving plans for concurrent dataset workers."""

from dataclasses import dataclass

from dataset_agent.core.instance_pickers.balancer import DistributionBalancer
from dataset_agent.dataset_source import get_source_dataset
from dataset_agent.models.dataset_generation_state_model import (
    DatasetState,
    DistributionBucket,
    DistributionState,
)
from settings import settings


@dataclass(frozen=True)
class WorkerGenerationPlan:
    """The immutable source-data scope and initial state for one worker."""

    worker_id: int
    class_labels: tuple[str, ...]
    initial_state: DatasetState


def _source_class_count(source_dataset) -> dict[str, int]:
    label_column = settings.source_dataset.label_col_name
    return source_dataset[label_column].value_counts().to_dict()


def get_worker_generation_plans() -> list[WorkerGenerationPlan]:
    """Build one non-overlapping source scope for every configured worker.

    Source quotas are calculated once across all labels before they are split
    into groups. This preserves the configured balanced or random distribution
    for the final merged dataset rather than balancing each group in isolation.
    """
    class_groups = settings.workflow.worker_class_groups
    if class_groups is None:
        raise RuntimeError("worker class groups must be configured before planning workers")

    source_dataset = get_source_dataset()
    source_settings = settings.source_dataset
    source_distribution = DistributionBalancer.from_settings(
        source_settings,
        target_size=settings.output_dataset.target_size,
        seed=42,
        dataset_class_count=_source_class_count(source_dataset),
    ).init_distribution()
    label_column = source_settings.label_col_name
    plans: list[WorkerGenerationPlan] = []

    for worker_id, class_group in enumerate(class_groups, start=1):
        # Pre-populating the state prevents the picker from selecting another
        # worker's classes while retaining its normal retry and sampling logic.
        input_distribution: DistributionState = {
            class_name: DistributionBucket(
                target=source_distribution[class_name].target,
                actual=0,
            )
            for class_name in class_group
        }
        worker_target_size = sum(bucket.target for bucket in input_distribution.values())
        if worker_target_size == 0:
            raise ValueError(
                f"worker {worker_id} has no assigned generation quota; "
                "adjust worker class groups or increase output_dataset.target_size"
            )

        remaining_input_indices = source_dataset.index[
            source_dataset[label_column].isin(class_group)
        ].tolist()
        plans.append(
            WorkerGenerationPlan(
                worker_id=worker_id,
                class_labels=tuple(class_group),
                initial_state=DatasetState(
                    target_size=worker_target_size,
                    index_to_generate=0,
                    remaining_input_indices=remaining_input_indices,
                    input_distribution=input_distribution,
                ),
            )
        )

    return plans