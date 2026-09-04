"""Private module that holds some path constant definition"""

from pathlib import Path
from typing import Literal

from app.settings import settings

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[3] / "output"

GENERATED_DATASET_PATH = (
    OUTPUT_DIRECTORY / f"{settings.output_dataset.name}.{settings.output_dataset.format}"
)

SPLIT_MANIFEST_PATH = OUTPUT_DIRECTORY / "split_manifest.csv"

JLENS_REPOSITORY = "neuronpedia/jacobian-lens"


def get_feature_dataset_path(
    jlens_model_name: str,
    pooling_type: str,
    output_format: Literal["NumPy", "Parquet"],
) -> Path:
    match output_format:
        case "NumPy":
            ext = ".npz"
        case "Parquet":
            ext = ".parquet"
        case _:
            raise ValueError("The output format must be either NumPy or Parquet")

    model_name = "-".join(jlens_model_name.split("/"))
    return OUTPUT_DIRECTORY / f"features_{model_name}_{pooling_type}{ext}"
