"""Rebuild the final dataset from augmented output and worker checkpoints.

Run from the project root with:
    python notebooks/merge_not_selected_indices.py
"""

import os
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a configured CSV or JSONL dataset."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported output format: {path.suffix}")


def _write_dataset(dataset: pd.DataFrame, path: Path) -> None:
    """Write a dataframe in the configured CSV or JSONL output format."""
    if path.suffix.lower() == ".csv":
        dataset.to_csv(path, index=False)
    elif path.suffix.lower() == ".jsonl":
        dataset.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")


def _get_remaining_indices(checkpoint_path: Path) -> list[int]:
    """Load the final unselected source indices from one graph checkpoint."""
    from dataset_agent.graph import build_and_compile_graph

    config = {"configurable": {"thread_id": checkpoint_path.stem}}
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_and_compile_graph(checkpointer=checkpointer)
        snapshot = graph.get_state(config)

    remaining_indices = snapshot.values.get("remaining_input_indices")
    if remaining_indices is None:
        raise RuntimeError(f"No completed state found in {checkpoint_path.name}")
    return remaining_indices


def main() -> None:
    """Create augmented plus not-selected original records in the final output file."""
    from app.settings import settings
    from dataset_agent.dataset_source import get_source_dataset

    print(get_source_dataset().shape)

    if settings.output_dataset.merge != "not_selected_indeces":
        raise ValueError(
            "Set output_dataset.merge to 'not_selected_indeces' before running this script."
        )

    output_directory = PROJECT_ROOT / "output"
    checkpoint_directory = output_directory / ".checkpoints"
    final_path = output_directory / (
        f"{settings.output_dataset.name}.{settings.output_dataset.format}"
    )
    snapshot_path = final_path.with_name(f"{final_path.stem}.augmented{final_path.suffix}")

    if not final_path.is_file():
        raise FileNotFoundError(f"Augmented dataset not found: {final_path}")

    checkpoint_paths = sorted(checkpoint_directory.glob("worker-*.sqlite"))
    if not checkpoint_paths:
        checkpoint_paths = [checkpoint_directory / "single-worker.sqlite"]

    missing_checkpoints = [path for path in checkpoint_paths if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(f"Checkpoint files not found: {missing_checkpoints}")

    remaining_input_indices = [
        index
        for checkpoint_path in checkpoint_paths
        for index in _get_remaining_indices(checkpoint_path)
    ]
    if len(remaining_input_indices) != len(set(remaining_input_indices)):
        raise RuntimeError("Worker checkpoints contain overlapping remaining source indices.")

    if not snapshot_path.exists():
        shutil.copy2(final_path, snapshot_path)
        print(f"Created augmented snapshot: {snapshot_path}")

    augmented_dataset = _read_dataset(snapshot_path)
    unused_source_dataset = get_source_dataset().loc[remaining_input_indices]
    merged_dataset = pd.concat([augmented_dataset, unused_source_dataset], ignore_index=True)

    temporary_path = final_path.with_name(
        f".{final_path.stem}.{uuid4().hex}.tmp{final_path.suffix}"
    )
    try:
        _write_dataset(merged_dataset, temporary_path)
        os.replace(temporary_path, final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Checkpoints read: {len(checkpoint_paths)}")
    print(f"Augmented records: {len(augmented_dataset)}")
    print(f"Original records added: {len(unused_source_dataset)}")
    print(f"Final records: {len(merged_dataset)}")
    print(f"Wrote: {final_path}")


if __name__ == "__main__":
    main()
