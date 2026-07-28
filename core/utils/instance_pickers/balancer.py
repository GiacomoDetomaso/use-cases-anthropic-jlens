from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

from core.models.dataset_generation_state_model import DistributionBucket, DistributionState

from core.utils.instance_pickers.distribution_calculator import (
    build_distribution_calculator,
)

if TYPE_CHECKING:
    from settings import DatasetSettings

DistributionStrategy = Literal["balanced", "random"] | Mapping[str, float]

class DistributionBalancer:
    def __init__(
        self, 
        class_labels: Sequence[str], 
        target_size: int, 
        strategy: DistributionStrategy, 
        seed: int | None = None,
        dataset_class_count: dict[str, int]=None
    ):
        if not class_labels:
            raise ValueError("class_labels cannot be empty")

        self.class_labels = list(class_labels)
        self.target_size = target_size
        self.strategy = strategy
        self._rng = random.Random(seed)
        self.dataset_class_count = dataset_class_count

    @classmethod
    def from_settings(
        cls, 
        dataset_settings: DatasetSettings, 
        target_size: int, 
        seed: int | None = None,
        dataset_class_count: dict[str, int]=None
    ):
        return cls(
            class_labels=dataset_settings.class_labels,
            target_size=target_size,
            strategy=dataset_settings.class_distribution,
            seed=seed,
            dataset_class_count=dataset_class_count
        )

    def init_distribution(self) -> DistributionState:
        targets = self._calculate_targets() 

        return {
            class_name: DistributionBucket(target=target, actual=0)
            for class_name, target in targets.items()
        }

    def pick_class(self, distribution: DistributionState) -> str:
        priorities = {
            class_name: distribution[class_name].target - distribution[class_name].actual
            for class_name in distribution
        }
        
        if not priorities:
            raise ValueError("Distribution cannot be empty")

        max_needed = max(priorities.values())
        if max_needed <= 0:
            raise ValueError("All classes already reached their targets")

        weighted_classes = [
            class_name
            for class_name, priority in priorities.items()
            if priority >= max_needed * 0.8
        ]

        return self._rng.choice(weighted_classes)

    def update_class_distribution(
        self, distribution: DistributionState, 
        class_name: str, 
        amount: int = 1
    ) -> DistributionState:
        if class_name not in distribution:
            raise KeyError(f"Class '{class_name}' is not in the distribution")

        updated_distribution = distribution.copy()
        bucket = updated_distribution[class_name]
        updated_distribution[class_name] = bucket.model_copy(update={"actual": bucket.actual + amount})

        return updated_distribution

    def _calculate_targets(self) -> dict[str, int]:
        if self.target_size < 0:
            raise ValueError("target_size cannot be negative")

        calculator = build_distribution_calculator(
            class_labels=set(self.class_labels),
            target_size=self.target_size,
            rng=self._rng,
            dataset_class_count=self.dataset_class_count,
        )

        match self.strategy:
            case "balanced":
                return calculator.balanced_targets()
            case "random":
                return calculator.random_targets()
            case isinstance(self.strategy, Mapping):
                return calculator.percentage_targets(self.strategy)
            case _:
                raise ValueError(f"Unsupported distribution strategy: {self.strategy}")
