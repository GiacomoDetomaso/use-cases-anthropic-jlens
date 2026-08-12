import json

from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SyntheticRecord:
    source: str
    target: str
    output: str

class DatasetWriter(ABC):
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.dataset: list[SyntheticRecord] = []

        output_path.mkdir(parents=True, exist_ok=True)
       
    def append(self, record: SyntheticRecord) -> None:
        self.dataset.append(record)

    @abstractmethod
    def serialize(self, start: int, stop: int) -> bool:
        pass

class JsonLDatasetWriter(DatasetWriter):
    def __init__(self, output_path: Path, file_name: str = "dataset.jsonl"):
        super().__init__(output_path)
        self.file_path = output_path / file_name

    def serialize(self, start: int, stop: int):
        if stop < start: 
            raise ValueError("Upper bound can't be lower than lower bound")

        if not 0 <= start <= stop <= len(self.dataset):
            raise IndexError("Invalid serialization bounds")

        with open(self.file_path, "a", encoding="utf-8") as f:
            for record in self.dataset[start:stop]:
                f.write(json.dumps(record.__dict__))
                f.write("\n")
