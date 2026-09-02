import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler  # type: ignore[import-untyped]
from loguru import logger
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

from app.settings import settings
from training.dataset._paths import GENERATED_DATASET_PATH, SPLITTED_DATASET_FOLDER


def build_splits() -> None:
    match settings.output_dataset.format:
        case "csv":
            df = pd.read_csv(GENERATED_DATASET_PATH)
        case "jsonl":
            df = pd.read_json(GENERATED_DATASET_PATH)
        case _:
            raise RuntimeError("No available dataset format")

    data, labels = df["text"], df["label_1"]
    numeric_labels = np.where(labels.to_numpy() == "benign", 0, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        numeric_labels,
        train_size=85,
        stratify=numeric_labels,
        random_state=settings.training.random_state,
    )

    minority_class_ratio: float | None = None

    match settings.training.train_sampling:
        case "50/50":
            minority_class_ratio = 1.0
        case "70/30":
            minority_class_ratio = 3 / 7
        case _:
            logger.info("No sampling strategy selected on the training set")

    if minority_class_ratio is not None:
        sampler = RandomUnderSampler(
            sampling_strategy=minority_class_ratio, random_state=settings.training.random_state
        )

        sampled_features, y_train = sampler.fit_resample(
            X=x_train.to_frame(name="text"),
            y=y_train,
        )
        x_train = sampled_features["text"]

    # Treat splits as pandas Series to allow a clean save to .csv files
    x_train = pd.Series(x_train)
    y_train = pd.Series(y_train)
    x_test = pd.Series(x_test)
    y_test = pd.Series(y_test)

    x_train.to_csv(SPLITTED_DATASET_FOLDER / "x_train.csv")
    y_train.to_csv(SPLITTED_DATASET_FOLDER / "y_train.csv")
    x_test.to_csv(SPLITTED_DATASET_FOLDER / "x_test.csv")
    y_test.to_csv(SPLITTED_DATASET_FOLDER / "y_test.csv")
