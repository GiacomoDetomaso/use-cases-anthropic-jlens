"""Invoke dataset-generation graphs and manage their persisted outputs.

Public Interfaces
-----------------
generate_dataset
    Run one dataset-generation graph synchronously.
generate_dataset_async
    Run one dataset-generation graph asynchronously.
generate_datasets_with_workers
    Run independently checkpointed asynchronous workers and merge their outputs.
"""

import asyncio
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pandas as pd
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from loguru import logger

from app.settings import settings
from dataset_agent.core.inference.inference_context_manager import (
    inference_environment,
)
from dataset_agent.core.writers.writers_builder import (
    build_dataset_writer,
    merge_dataset_files,
)
from dataset_agent.dataset_source import get_source_dataset
from dataset_agent.graph import (
    build_and_compile_graph,
    get_graph_initial_state,
)
from dataset_agent.worker_generation_plan import (
    WorkerGenerationPlan,
    build_worker_generation_plans,
)


def _prepare_single_worker_run():
    """Prepare state and filesystem resources for a single-worker graph run.

    Returns
    -------
    tuple
        Initial graph state, output directory, checkpoint path, and LangGraph
        invocation configuration, respectively.
    """
    initial_state = get_graph_initial_state()
    output_directory = _output_directory_path()
    checkpoint_path = output_directory / ".checkpoints" / "single-worker.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_checkpoint(checkpoint_path)

    if not settings.workflow.resume:
        (output_directory / _dataset_file_name()).unlink(missing_ok=True)

    config = {"configurable": {"thread_id": "single-worker"}}

    return initial_state, output_directory, checkpoint_path, config


def _output_directory_path() -> Path:
    """Return the repository-local directory for generated dataset files.

    Returns
    -------
    pathlib.Path
        The ``output`` directory at the project root.
    """
    return Path(__file__).resolve().parents[2] / "output"


def _dataset_file_name() -> str:
    """Build the configured dataset file name.

    Returns
    -------
    str
        A file name composed from the configured dataset name and format.
    """
    return f"{settings.output_dataset.name}.{settings.output_dataset.format}"


def _is_serialized_state(state: dict, record_count: int) -> bool:
    """Determine whether a checkpoint matches a fully persisted shard.

    Parameters
    ----------
    state : dict
        Checkpoint state values to inspect.
    record_count : int
        Number of records currently persisted in the shard.

    Returns
    -------
    bool
        ``True`` when the checkpoint has completed the persisted records and
        contains no transient generation values.
    """
    transient_fields = (
        "source",
        "target",
        "generated_prompt",
        "regenerated_prompt",
        "validation_output",
    )
    return (
        state.get("index_to_generate") == record_count
        and state.get("last_checkpoint_index") == record_count
        and all(state.get(field) is None for field in transient_fields)
    )


def _restore_resumable_sync_state(graph, config: dict, output_directory: Path, file_name: str):
    """Restore a compatible synchronous graph checkpoint when configured.

    Parameters
    ----------
    graph
        Compiled LangGraph instance with synchronous state methods.
    config : dict
        LangGraph thread configuration used to locate checkpoints.
    output_directory : pathlib.Path
        Directory containing the persisted dataset shard.
    file_name : str
        Name of the persisted dataset shard.

    Returns
    -------
    dict or None
        A resumed graph configuration, or ``None`` when no resumption applies.

    Raises
    ------
    RuntimeError
        If shard data exists without a checkpoint that matches its record count.
    """
    if not settings.workflow.resume:
        return None

    record_count = len(build_dataset_writer(output_directory, file_name).dataset)
    for snapshot in graph.get_state_history(config):
        if _is_serialized_state(snapshot.values, record_count):
            return graph.update_state(snapshot.config, {}, as_node="save")

    if record_count:
        raise RuntimeError("Shard data exists but no matching state checkpoint was found")
    return None


async def _restore_resumable_async_state(
    graph, config: dict, output_directory: Path, file_name: str
):
    """Restore a compatible asynchronous graph checkpoint when configured.

    Parameters
    ----------
    graph
        Compiled LangGraph instance with asynchronous state methods.
    config : dict
        LangGraph thread configuration used to locate checkpoints.
    output_directory : pathlib.Path
        Directory containing the persisted dataset shard.
    file_name : str
        Name of the persisted dataset shard.

    Returns
    -------
    dict or None
        A resumed graph configuration, or ``None`` when no resumption applies.

    Raises
    ------
    RuntimeError
        If shard data exists without a checkpoint that matches its record count.
    """
    if not settings.workflow.resume:
        return None

    record_count = len(build_dataset_writer(output_directory, file_name).dataset)
    async for snapshot in graph.aget_state_history(config):
        if _is_serialized_state(snapshot.values, record_count):
            return await graph.aupdate_state(snapshot.config, {}, as_node="save")

    if record_count:
        raise RuntimeError("Shard data exists but no matching state checkpoint was found")
    return None


def _clear_checkpoint(checkpoint_path: Path) -> None:
    """Remove a checkpoint and SQLite sidecar files when not resuming.

    Parameters
    ----------
    checkpoint_path : pathlib.Path
        Base path of the worker's SQLite checkpoint database.

    Returns
    -------
    None
    """
    if settings.workflow.resume:
        return

    for path in (
        checkpoint_path,
        checkpoint_path.with_name(f"{checkpoint_path.name}-shm"),
        checkpoint_path.with_name(f"{checkpoint_path.name}-wal"),
    ):
        path.unlink(missing_ok=True)


async def _generate_worker(
    plan: WorkerGenerationPlan, run_directory: Path
) -> tuple[Path, list[int]]:
    """Generate one independently checkpointed worker shard.

    Parameters
    ----------
    plan : WorkerGenerationPlan
        Worker-specific state, class assignment, and numeric identifier.
    run_directory : pathlib.Path
        Directory in which to write the worker's output shard.

    Returns
    -------
    tuple[pathlib.Path, list[int]]
        Path to the generated worker shard and source indices left unused by it.
    """
    worker_name = f"worker-{plan.worker_id:02d}"

    with logger.contextualize(worker=worker_name):
        logger.info(
            "Generating {} records from source classes: {}",
            plan.initial_state.target_size,
            list(plan.class_labels),
        )

        output_file_name = f"worker-{plan.worker_id:02d}.{settings.output_dataset.format}"

        checkpoint_path = run_directory.parent / ".checkpoints" / f"{worker_name}.sqlite"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        _clear_checkpoint(checkpoint_path)
        config = {"configurable": {"thread_id": worker_name}}

        async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
            graph = build_and_compile_graph(
                target_size=plan.initial_state.target_size,
                output_path=run_directory,
                output_file_name=output_file_name,
                seed=42 + plan.worker_id,
                checkpointer=checkpointer,
            )
            resume_config = await _restore_resumable_async_state(
                graph, config, run_directory, output_file_name
            )
            await graph.ainvoke(
                None if resume_config else plan.initial_state,
                resume_config or config,
            )
            final_state = await graph.aget_state(config)

    return run_directory / output_file_name, final_state.values["remaining_input_indices"]


def _compose_final_dataset(
    augmented_files: list[Path], remaining_input_indices: list[int] | None = None
) -> None:
    """Atomically compose augmented output with configured source examples.

    Parameters
    ----------
    augmented_files : list[pathlib.Path]
        Generated shard paths in their intended output order.
    remaining_input_indices : list[int] or None
        Source indices not consumed during generation. Required only when
        ``output_dataset.merge`` is ``not_selected_indeces``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the configured merge mode requires checkpoint state that is absent.
    """
    output_directory = _output_directory_path()
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / _dataset_file_name()
    temporary_path = output_directory / (f".{final_path.stem}.{uuid4().hex}.tmp{final_path.suffix}")

    try:
        merge_mode = settings.output_dataset.merge
        if merge_mode == "none":
            merge_dataset_files(augmented_files, temporary_path)
        else:
            augmented_dataset = _read_dataset_files(augmented_files)
            source_dataset = get_source_dataset()
            if merge_mode == "all":
                original_dataset = source_dataset
                combined_dataset = pd.concat(
                    [original_dataset, augmented_dataset], ignore_index=True
                )
            else:
                if remaining_input_indices is None:
                    raise ValueError(
                        "Missing remaining source indices for configured dataset merge"
                    )
                original_dataset = source_dataset.loc[remaining_input_indices]
                combined_dataset = pd.concat(
                    [augmented_dataset, original_dataset], ignore_index=True
                )
            _write_dataset_file(combined_dataset, temporary_path)
        os.replace(temporary_path, final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_dataset_files(files: list[Path]) -> pd.DataFrame:
    """Load homogeneous generated shards into one tabular dataset."""
    if not files:
        raise ValueError("At least one CSV or JSONL augmented shard is required")

    if any(file.suffix.lower() != files[0].suffix.lower() for file in files):
        raise ValueError("Augmented shards must use the same format")

    suffix = files[0].suffix.lower()
    if suffix == ".csv":
        return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    if suffix == ".jsonl":
        return pd.concat([pd.read_json(file, lines=True) for file in files], ignore_index=True)
    raise ValueError(f"Unsupported dataset format: {files[0].name}")


def _write_dataset_file(dataset: pd.DataFrame, path: Path) -> None:
    """Write a combined dataset in the configured output format."""
    if path.suffix.lower() == ".csv":
        dataset.to_csv(path, index=False)
    elif path.suffix.lower() == ".jsonl":
        dataset.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported dataset format: {path.name}")


def generate_dataset() -> None:
    """Run a single dataset-generation graph synchronously.

    The graph uses the configured dataset settings and writes its output to the
    configured output directory. When resumption is disabled, prior output and
    checkpoint files for the single worker are removed before execution.

    Returns
    -------
    None
        The generated dataset is persisted to disk.

    Raises
    ------
    RuntimeError
        If resumption is enabled and shard data has no matching checkpoint.
    """
    initial_state, output_directory, checkpoint_path, config = _prepare_single_worker_run()

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_and_compile_graph(
            output_path=output_directory,
            checkpointer=checkpointer,
        )
        with inference_environment():
            resume_config = _restore_resumable_sync_state(
                graph, config, output_directory, _dataset_file_name()
            )
            graph.invoke(None if resume_config else initial_state, resume_config or config)
            final_state = graph.get_state(config)

    _compose_final_dataset(
        [output_directory / _dataset_file_name()],
        final_state.values["remaining_input_indices"],
    )


async def generate_dataset_async() -> None:
    """Run a single dataset-generation graph asynchronously.

    The graph uses the configured dataset settings and writes its output to the
    configured output directory. When resumption is disabled, prior output and
    checkpoint files for the single worker are removed before execution.

    Returns
    -------
    None
        The generated dataset is persisted to disk.

    Raises
    ------
    RuntimeError
        If resumption is enabled and shard data has no matching checkpoint.
    """
    initial_state, output_directory, checkpoint_path, config = _prepare_single_worker_run()

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_and_compile_graph(
            output_path=output_directory,
            checkpointer=checkpointer,
        )
        with inference_environment():
            resume_config = await _restore_resumable_async_state(
                graph, config, output_directory, _dataset_file_name()
            )
            await graph.ainvoke(
                None if resume_config else initial_state,
                resume_config or config,
            )
            final_state = await graph.aget_state(config)

    _compose_final_dataset(
        [output_directory / _dataset_file_name()],
        final_state.values["remaining_input_indices"],
    )


async def generate_datasets_with_workers() -> None:
    """Generate dataset shards concurrently and merge them into one dataset.

    Worker plans are derived from the configured class distribution. Each worker
    writes an independently checkpointed shard, which is merged atomically after
    every worker finishes successfully.

    Returns
    -------
    None
        The merged dataset is persisted to the configured output directory.

    Raises
    ------
    Exception
        Propagates worker-generation or output-merge failures.
    """
    plans = build_worker_generation_plans()
    output_directory = _output_directory_path()
    run_directory = output_directory / ".runs"

    if not settings.workflow.resume:
        shutil.rmtree(run_directory, ignore_errors=True)
        shutil.rmtree(output_directory / ".checkpoints", ignore_errors=True)
    run_directory.mkdir(parents=True, exist_ok=True)

    with inference_environment():
        async with asyncio.TaskGroup() as task_group:
            tasks = [
                task_group.create_task(_generate_worker(plan, run_directory)) for plan in plans
            ]

    worker_results = [task.result() for task in tasks]
    _compose_final_dataset(
        [worker_file for worker_file, _ in worker_results],
        [index for _, indices in worker_results for index in indices],
    )
    logger.info("Merged {} worker outputs into {}", len(plans), _dataset_file_name())
