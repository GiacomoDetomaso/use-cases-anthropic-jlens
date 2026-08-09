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
        self.dataset.append(record.__dict__)

    @abstractmethod
    def serialize(self, start: int, stop: int) -> bool:
        pass

class JsonLDatasetWriter(DatasetWriter):
    def __init__(self, output_path):
        super().__init__(output_path)

    def serialize(self, start: int, stop: int):
        if stop < start: 
            raise ValueError("Upper bound can't be lower than lower bound")

        dataset_range_lst = list(range(self.dataset))

        if start in dataset_range_lst and stop in dataset_range_lst:
            with open(self.output_path, "a", encoding="utf-8") as f:
                for record in self.dataset[start:stop]:
                    f.write(record.__dict__)
                    f.write("\n")
        else:
            raise IndexError(f"Lower or upper bounds are not in the dataset range.\nstart={start}\nstop={stop}\nrange=(0, {len(self.dataset)})")        
