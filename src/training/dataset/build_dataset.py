"""Build and load the feature dataset used for training."""

from pathlib import Path

import pandas as pd
from loguru import logger

from training.dataset._paths import FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT
from training.dataset.feature import extract_jlens_features, load_feature_dataset
from training.dataset.splitter import EXAMPLE_ID_COLUMN, build_splits


def _find_feature_dataset_path() -> Path | None:
    for extension in (".npz", ".parquet"):
        feature_path = FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT.with_suffix(extension)
        if feature_path.is_file():
            return feature_path
    return None


def build(remove_example_id_col=False) -> pd.DataFrame:
    """Build the split manifest and return the feature dataset in tabular form."""
    logger.info("Building split manifest")
    build_splits()

    feature_path = _find_feature_dataset_path()

    if feature_path is None:
        logger.info("No extracted feature dataset found; starting J-lens feature extraction")
        extract_jlens_features()
        feature_path = _find_feature_dataset_path()
    else:
        logger.info("Using existing extracted feature dataset: {}", feature_path)

    if feature_path is None:
        raise RuntimeError("Feature extraction did not create an output dataset")

    fd = load_feature_dataset(feature_path)
    logger.info("Loaded feature dataset with {} rows and {} columns", *fd.shape)

    if remove_example_id_col:
        fd = fd.drop(columns=[EXAMPLE_ID_COLUMN])
        logger.info("Removed example ID column from feature dataset")

    return fd
