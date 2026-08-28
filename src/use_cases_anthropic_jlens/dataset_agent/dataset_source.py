"""
This module loads the CSV datasets as pandas DataFrames, it stores their instances
as global singletons that can be accessed everywhere in the application:
  - `source`: the customer-support dataset used as generation input.
  - `target_transformation_examples_dataset`: reference examples for each
    target injection class, used to guide the transformation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Protocol

import pandas as pd
from loguru import logger

from use_cases_anthropic_jlens.settings import settings


class _DatasetConfig(Protocol):
    input_source: str
    class_labels: list[str]
    data_col_name: str
    label_col_name: str


def _read_dataset(input_source: str) -> pd.DataFrame:
    suffix = Path(input_source).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_source)
    if suffix == ".parquet":
        return pd.read_parquet(input_source)

    raise ValueError(
        f"Unsupported dataset format '{suffix or '<none>'}' for '{input_source}'. "
        "Supported formats are CSV and Parquet."
    )


def _load_and_filter(config: _DatasetConfig) -> pd.DataFrame:
    df = _read_dataset(config.input_source)

    data_col_name, label = config.data_col_name, config.label_col_name
    keep_cols = [data_col_name, label]

    df = df[keep_cols].copy()

    try:
        dataset_labels_set = set(df[label].unique().tolist())
        user_defined_labels_set = set(config.class_labels)

        diff_set = user_defined_labels_set.difference(dataset_labels_set)

        if len(diff_set) == 0:
            is_in = df[label].isin(config.class_labels)
            df = df[is_in]
        else:
            raise KeyError(
                f"Some label [{diff_set}] are not in the actual dataset, falling back to all labels"
            )
    except KeyError:
        logger.info("Label column '{}' not found while filtering by class_labels", label)

    return df


@lru_cache(maxsize=1)
def _load_source_dataset() -> pd.DataFrame:
    return _load_and_filter(settings.source_dataset)


@lru_cache(maxsize=1)
def _load_target_transformation_examples_dataset() -> pd.DataFrame:
    return _load_and_filter(settings.target_transformation_examples_dataset)


def get_source_dataset() -> pd.DataFrame:
    """Return a mutable copy of the cached source dataset.

    The module-level cache stays read-only; callers get their own copy
    so mutations never leak back into the shared singleton.
    """
    return _load_source_dataset().copy()


def get_target_transformation_examples_dataset() -> pd.DataFrame:
    """Return a mutable copy of the cached target transformation examples dataset.

    The module-level cache stays read-only; callers get their own copy
    so mutations never leak back into the shared singleton.
    """
    return _load_target_transformation_examples_dataset().copy()


source = _load_source_dataset()
target_transformation_examples_dataset = _load_target_transformation_examples_dataset()
