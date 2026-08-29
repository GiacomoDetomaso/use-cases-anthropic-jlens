"""Writer construction and shard-merging interface for generation services."""

from pathlib import Path

from dataset_agent.core.writers.writers import (
    CsvDatasetWriter,
    DatasetWriter,
    JsonLDatasetWriter,
)

_WRITER_CLASSES = {
    ".csv": CsvDatasetWriter,
    ".jsonl": JsonLDatasetWriter,
}


def build_dataset_writer(output_path: Path, file_name: str) -> DatasetWriter:
    """Build the writer selected by a dataset filename extension.

    Parameters
    ----------
    output_path : pathlib.Path
        Directory in which the dataset file is stored.
    file_name : str
        Dataset filename ending in ``.csv`` or ``.jsonl``.

    Returns
    -------
    DatasetWriter
        CSV or JSONL writer appropriate for ``file_name``.

    Raises
    ------
    ValueError
        If ``file_name`` has an unsupported extension.
    """
    try:
        writer_class = _WRITER_CLASSES[Path(file_name).suffix.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset format: {file_name}") from error

    return writer_class(output_path, file_name)


def merge_dataset_files(worker_files: list[Path], destination: Path) -> None:
    """Merge worker shards using the writer for the destination format.

    Parameters
    ----------
    worker_files : list[pathlib.Path]
        Ordered worker shard files to merge.
    destination : pathlib.Path
        Final dataset path, ending in ``.csv`` or ``.jsonl``.

    Raises
    ------
    ValueError
        If no worker files are supplied, formats differ, or the destination
        extension is unsupported.
    """
    if not worker_files:
        raise ValueError("At least one worker file is required for merging")

    extensions = {worker_file.suffix.lower() for worker_file in worker_files}
    if extensions != {destination.suffix.lower()}:
        raise ValueError("Worker shards and destination must use the same format")

    try:
        writer_class = _WRITER_CLASSES[destination.suffix.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset format: {destination.name}") from error

    writer_class.merge_files(worker_files, destination)
