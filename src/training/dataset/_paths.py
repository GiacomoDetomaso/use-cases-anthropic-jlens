"""Private module that holds some path constant definition"""

from pathlib import Path

from app.settings import settings

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[3] / "output"

GENERATED_DATASET_PATH = (
    OUTPUT_DIRECTORY / f"{settings.output_dataset.name}.{settings.output_dataset.format}"
)

SPLIT_MANIFEST_PATH = OUTPUT_DIRECTORY / "split_manifest.csv"

FEATURE_EXTRACTED_BASE_FILE_PATH_NO_EXT = OUTPUT_DIRECTORY / "features"

JLENS_REPOSITORY = "neuronpedia/jacobian-lens"
