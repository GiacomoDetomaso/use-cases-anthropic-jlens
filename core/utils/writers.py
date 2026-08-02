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
        self.dataset: list[SyntheticRecord] = {}

        output_path.mkdir(parents=True, exist_ok=True)
       
    def append(self, record: SyntheticRecord) -> None:
        self.dataset.append(record.__dict__)

    @abstractmethod
    def serialize(self) -> bool:
        pass

class JsonLDatasetWriter(DatasetWriter):
    def __init__(self, output_path):
        super().__init__(output_path)

    def serialize(self):
        with open(self.output_path, "a", encoding="utf-8") as f:
            for record in self.dataset:
                f.write(record.__dict__)
                f.write("\n")        
