import pandas as pd

from dataset_agent.models.dataset_generation_io_models import InputAttackModel, InputDatasetModel
from dataset_agent.models.dataset_generation_state_model import DatasetState
from dataset_agent.core.instance_pickers.balancer import DistributionBalancer
from dataset_agent.core.instance_pickers.sampler import DatasetClassSampler
from settings import DatasetSettings, settings


class PickerInputDatasetNode:
    def __init__(self, dataset: pd.DataFrame, dataset_settings: DatasetSettings, seed: int | None = None):
        self.dataset_settings = dataset_settings
        self.seed = seed

        self.sampler = DatasetClassSampler(
            dataset=dataset,
            data_col_name=dataset_settings.data_col_name,
            label_col_name=dataset_settings.label_col_name,
            seed=seed,
        )

        self._dataset_class_count = (
            dataset[dataset_settings.label_col_name]
                .value_counts()
                .to_dict()
        )

        self.balancer = DistributionBalancer.from_settings(
            dataset_settings,
            target_size=settings.output_dataset.target_size,
            seed=seed,
            dataset_class_count=self._dataset_class_count,
        )

    def __call__(self, state: DatasetState) -> DatasetState:
        input_distribution = state.input_distribution or self.balancer.init_distribution()

        class_name = self.balancer.pick_class(input_distribution)
        selected_index, row = self.sampler.sample_one(class_name, state.remaining_input_indices)

        return state.model_copy(update={
            "source": InputDatasetModel(
                original_prompt=row[self.sampler.data_col_name],
                original_intent=row[self.sampler.label_col_name],
            ),
            "remaining_input_indices": [index for index in state.remaining_input_indices if index != selected_index],
            "input_distribution": self.balancer.update_class_distribution(input_distribution, class_name),
        })


class PickerTargetDatasetNode:
    def __init__(self, dataset: pd.DataFrame, dataset_settings: DatasetSettings, target_size: int, examples_count: int, seed: int | None = None):
        self.examples_count = examples_count
        self.balancer = DistributionBalancer.from_settings(dataset_settings, target_size=target_size, seed=seed)
        self.sampler = DatasetClassSampler(
            dataset=dataset,
            data_col_name=dataset_settings.data_col_name,
            label_col_name=dataset_settings.label_col_name,
            seed=seed,
        )

    def __call__(self, state: DatasetState) -> DatasetState:
        target_distribution = state.target_distribution or self.balancer.init_distribution()
        class_name = self.balancer.pick_class(target_distribution)
        examples = self.sampler.sample_many(class_name, self.examples_count)

        target_examples = "\n\n".join(
            f"Example {index + 1}: {row[self.sampler.data_col_name]}"
            for index, row in enumerate(examples)
        )

        return state.model_copy(update={
            "target": InputAttackModel(
                target_intent=class_name,
                target_examples=target_examples,
            ),
            "target_distribution": self.balancer.update_class_distribution(target_distribution, class_name),
        })
