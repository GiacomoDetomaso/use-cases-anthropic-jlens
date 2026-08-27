"""Durable dataset writers for CSV and JSONL worker shards."""

import csv
from io import TextIOWrapper
import json
import os
import shutil

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, asdict, fields


@dataclass
class SyntheticRecord:
    """A generated dataset record.

    Parameters
    ----------
    source : str
        Source example used to create the record.
    target : str
        Target transformation example or category.
    output : str
        Generated prompt accepted by the validation stage.
    """

    source: str
    target: str
    output: str


class DatasetWriter(ABC):
    """Base class for resumable dataset shard writers.

    Parameters
    ----------
    output_path : pathlib.Path
        Directory that contains the dataset shard.
    file_name : str
        Name of the dataset shard within ``output_path``.

    Attributes
    ----------
    dataset : list[SyntheticRecord]
        Records loaded from disk and appended during the current run.
    file_path : pathlib.Path
        Full path of the dataset shard.
    """

    def __init__(self, output_path: Path, file_name: str):
        self.output_path = output_path
        self.dataset: list[SyntheticRecord] = []
        self.file_name = file_name

        output_path.mkdir(parents=True, exist_ok=True)
        self.file_path = output_path / file_name

        if self.file_path.exists():
            with open(self.file_path, newline="", encoding="utf-8") as file:
                self.dataset = self._reload_resumed_dataset(file)

    def append(self, record: SyntheticRecord) -> None:
        """Append a record to the in-memory dataset.

        Parameters
        ----------
        record : SyntheticRecord
            Record to append.
        """
        self.dataset.append(record)

    def append_at(self, index: int, record: SyntheticRecord) -> bool:
        """Append a record at an expected position without duplication.

        Parameters
        ----------
        index : int
            Zero-based position represented by the graph state.
        record : SyntheticRecord
            Record expected at ``index``.

        Returns
        -------
        bool
            ``True`` when the record was appended; ``False`` when the same
            record already exists at ``index``.

        Raises
        ------
        ValueError
            If an existing record at ``index`` differs from ``record``.
        IndexError
            If the position would introduce a gap in the dataset.
        """
        if index == len(self.dataset):
            self.dataset.append(record)
            return True

        if 0 <= index < len(self.dataset):
            if self.dataset[index] != record:
                raise ValueError(f"Stored record {index + 1} differs from resumed state")
            return False

        raise IndexError(f"Cannot append record {index + 1} after {len(self.dataset)} records")

    def serialize(self, start: int, stop: int) -> None:
        """Persist a contiguous in-memory record range to the shard.

        Parameters
        ----------
        start : int
            Inclusive start index of records to write.
        stop : int
            Exclusive end index of records to write.

        Raises
        ------
        ValueError
            If ``stop`` precedes ``start``.
        IndexError
            If the requested range is outside the in-memory dataset.
        """
        if stop < start:
            raise ValueError("Upper bound can't be lower than lower bound")

        if not 0 <= start <= stop <= len(self.dataset):
            raise IndexError("Invalid serialization bounds")

        with open(self.file_path, "a", newline="", encoding="utf-8") as file:
            self._append_records(file, self.dataset[start:stop])
            file.flush()
            os.fsync(file.fileno())

    @abstractmethod
    def _append_records(self, file: TextIOWrapper, records: list[SyntheticRecord]) -> None:
        """Write records in the subclass-specific format.

        Parameters
        ----------
        file : io.TextIOWrapper
            Open shard file in append mode.
        records : list[SyntheticRecord]
            Records selected for persistence.
        """
        pass

    @abstractmethod
    def _reload_resumed_dataset(self, file: TextIOWrapper) -> list[SyntheticRecord]:
        """Load records from an existing shard.

        Parameters
        ----------
        file : io.TextIOWrapper
            Open shard file in read mode.

        Returns
        -------
        list[SyntheticRecord]
            Records reconstructed from the shard.
        """
        pass

    @classmethod
    @abstractmethod
    def merge_files(cls, worker_files: list[Path], destination: Path) -> None:
        """Merge homogeneous worker shards into one dataset file.

        Parameters
        ----------
        worker_files : list[pathlib.Path]
            Ordered shard paths to merge.
        destination : pathlib.Path
            Output file for the combined dataset.
        """
        pass


class JsonLDatasetWriter(DatasetWriter):
    """Dataset writer that persists one JSON object per line."""

    def __init__(self, output_path: Path, file_name: str = "dataset.jsonl"):
        super().__init__(output_path, file_name)

    def _reload_resumed_dataset(self, file: TextIOWrapper) -> list[SyntheticRecord]:
        return [
            SyntheticRecord(**json.loads(line))
            for line in file
            if line.strip()
        ]

    def _append_records(self, file: TextIOWrapper, records: list[SyntheticRecord]) -> None:
        for record in records:
            file.write(json.dumps(asdict(record)))
            file.write("\n")

    @classmethod
    def merge_files(cls, worker_files: list[Path], destination: Path) -> None:
        with open(destination, "w", encoding="utf-8") as output:
            for worker_file in worker_files:
                with open(worker_file, encoding="utf-8") as input_file:
                    shutil.copyfileobj(input_file, output)


class CsvDatasetWriter(DatasetWriter):
    """Dataset writer that stores structured source and target cells as JSON."""
    
    def __init__(self, output_path: Path, file_name: str = "dataset.csv"):
        super().__init__(output_path, file_name)

    def _reload_resumed_dataset(self, file: TextIOWrapper) -> list[SyntheticRecord]:
        return [
            SyntheticRecord(
                source=json.loads(row["source"]),
                target=json.loads(row["target"]),
                output=row["output"],
            )
            for row in csv.DictReader(file)
        ]

    def _append_records(self, file: TextIOWrapper, records: list[SyntheticRecord]) -> None:
        writer = csv.DictWriter(
            file,
            fieldnames=[field.name for field in fields(SyntheticRecord)],
        )

        if file.tell() == 0:
            writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "source": json.dumps(record.source),
                    "target": json.dumps(record.target),
                    "output": record.output,
                }
            )

    @classmethod
    def merge_files(cls, worker_files: list[Path], destination: Path) -> None:
        with open(destination, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=[field.name for field in fields(SyntheticRecord)],
            )
            writer.writeheader()
            for worker_file in worker_files:
                with open(worker_file, newline="", encoding="utf-8") as input_file:
                    writer.writerows(csv.DictReader(input_file))