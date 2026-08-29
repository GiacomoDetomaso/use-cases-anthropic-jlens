import random
from collections.abc import Sequence

import pandas as pd


class DatasetClassSampler:
    def __init__(
        self,
        dataset: pd.DataFrame,
        data_col_name: str,
        label_col_name: str,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.data_col_name = data_col_name
        self.label_col_name = label_col_name
        self._rng = random.Random(seed)

    def sample_one(
        self, class_name: str, candidate_indices: Sequence[int] | None = None
    ) -> tuple[int, pd.Series]:
        candidates = self._class_candidates(class_name, candidate_indices)
        if candidates.empty:
            raise ValueError(f"No records available for class '{class_name}'")

        index = self._rng.choice(candidates.index.tolist())
        return index, candidates.loc[index]

    def sample_many(self, class_name: str, count: int) -> list[pd.Series]:
        if count <= 0:
            raise ValueError("count must be greater than zero")

        candidates = self._class_candidates(class_name)
        if candidates.empty:
            raise ValueError(f"No records available for class '{class_name}'")

        replace = len(candidates) < count
        indices = (
            self._rng.choices(candidates.index.tolist(), k=count)
            if replace
            else self._rng.sample(candidates.index.tolist(), count)
        )
        return [candidates.loc[index] for index in indices]

    def _candidate_dataset(self, candidate_indices: Sequence[int] | None = None) -> pd.DataFrame:
        if candidate_indices is None:
            return self.dataset
        return self.dataset.loc[list(candidate_indices)]

    def _class_candidates(
        self, class_name: str, candidate_indices: Sequence[int] | None = None
    ) -> pd.DataFrame:
        dataset = self._candidate_dataset(candidate_indices)
        return dataset[dataset[self.label_col_name] == class_name]
