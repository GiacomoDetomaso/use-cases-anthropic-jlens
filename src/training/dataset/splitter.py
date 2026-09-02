import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler  # type: ignore[import-untyped]
from loguru import logger
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from app.settings import settings
from training.dataset._paths import GENERATED_DATASET_PATH, SPLIT_MANIFEST_PATH

EXAMPLE_ID_COLUMN = "example_id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SPLIT_COLUMN = "split"
TRAINING_SELECTED_COLUMN = "selected_for_training"


def _select_training_example_ids(train_rows: pd.DataFrame) -> pd.Series:
    """
    Return a Series of rows id of the input DataFrame selected for the training process.

    Parameters
    ----------
    train_rows : pd.DataFrame
        DataFrame with rows eligible for model training

    Returns
    -------
    pd.Series

    A series of ids taken from the `EXAMPLE_ID_COLUMN` of `train_rows`. It indicates
    which rows of `train_rows` can be used for training.
    """
    minority_class_ratio: float | None = None

    match settings.training.train_sampling:
        case "50/50":
            minority_class_ratio = 1.0
        case "70/30":
            minority_class_ratio = 3 / 7
        case _:
            logger.info("No sampling strategy selected on the training set")

    if minority_class_ratio is None:
        return train_rows[EXAMPLE_ID_COLUMN]

    sampler = RandomUnderSampler(
        sampling_strategy=minority_class_ratio, random_state=settings.training.random_state
    )
    sampled_rows, _ = sampler.fit_resample(train_rows, train_rows[LABEL_COLUMN])
    return sampled_rows[EXAMPLE_ID_COLUMN]


def build_splits() -> None:
    """
    It take the dataset stored at the specified path in the `training.yml`
    config file and:

    - Checks if it matches the required structure: it should contain the column text and label_1
    - Creates a manifest, in the form a .csv file associating at each row a unique id, the split (train, test) and whether
    the row should be used for training (if a random undersample is performed not all rows under the train split are eligible)
    """
    match settings.output_dataset.format:
        case "csv":
            df = pd.read_csv(GENERATED_DATASET_PATH)
        case "jsonl":
            df = pd.read_json(GENERATED_DATASET_PATH)
        case _:
            raise RuntimeError("No available dataset format")

    required_columns = {TEXT_COLUMN, "label_1"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Generated dataset is missing columns: {sorted(missing_columns)}")

    rows = pd.DataFrame(
        {
            EXAMPLE_ID_COLUMN: np.arange(len(df)),
            TEXT_COLUMN: df[TEXT_COLUMN].astype(str),
            LABEL_COLUMN: np.where(df["label_1"] == "benign", 0, 1),
        }
    )

    train_rows, test_rows = train_test_split(
        rows,
        train_size=0.85,
        stratify=rows[LABEL_COLUMN],
        random_state=settings.training.random_state,
    )

    selected_training_ids = set(_select_training_example_ids(train_rows))

    train_rows = train_rows.assign(
        **{
            SPLIT_COLUMN: "train",
            # Needed because if RandomUndersampling is performed not all rows are selected for training
            TRAINING_SELECTED_COLUMN: lambda frame: frame[EXAMPLE_ID_COLUMN].isin(
                selected_training_ids
            ),
        }
    )
    test_rows = test_rows.assign(**{SPLIT_COLUMN: "test", TRAINING_SELECTED_COLUMN: False})
    manifest = pd.concat([train_rows, test_rows], ignore_index=True)

    SPLIT_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(SPLIT_MANIFEST_PATH, index=False)
