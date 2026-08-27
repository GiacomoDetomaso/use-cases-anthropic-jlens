from operator import imod
import os
import shutil
import asyncio

from loguru import logger
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from dataset_agent.graph import build_and_compile_graph, get_graph_initial_state
from dataset_agent.core.inference.inference_context_manager import inference_environment
from dataset_agent.core.writers.writers_builder import (
    build_dataset_writer,
    merge_dataset_files,
)
from dataset_agent.worker_generation_plan import (
    WorkerGenerationPlan,
    build_worker_generation_plans,
)
from settings import settings

def generate_dataset() -> None:
    """
    Primitive to run the graph in synchronous mode
    """
    initial_state = get_graph_initial_state()
    output_directory = _output_directory()
    checkpoint_path = output_directory / ".checkpoints" / "single-worker.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_checkpoint(checkpoint_path)
    if not settings.workflow.resume:
        (output_directory / _dataset_file_name()).unlink(missing_ok=True)
    config = {"configurable": {"thread_id": "single-worker"}}

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


async def generate_dataset_async() -> None:
    """
    Primitive to run the graph in asynchronous mode
    """
    initial_state = get_graph_initial_state()
    output_directory = _output_directory()
    checkpoint_path = output_directory / ".checkpoints" / "single-worker.sqlite"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _clear_checkpoint(checkpoint_path)
    if not settings.workflow.resume:
        (output_directory / _dataset_file_name()).unlink(missing_ok=True)
    config = {"configurable": {"thread_id": "single-worker"}}
    
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

def _output_directory() -> Path:
    return Path(__file__).resolve().parent.parent / "output"


def _dataset_file_name() -> str:
    return f"dataset.{settings.output_dataset.format}"


def _is_serialized_state(state: dict, record_count: int) -> bool:
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
    if not settings.workflow.resume:
        return None

    record_count = len(build_dataset_writer(output_directory, file_name).dataset)
    for snapshot in graph.get_state_history(config):
        if _is_serialized_state(snapshot.values, record_count):
            return graph.update_state(snapshot.config, {}, as_node="save")

    if record_count:
        raise RuntimeError("Shard data exists but no matching state checkpoint was found")
    return None


async def _restore_resumable_async_state(graph, config: dict, output_directory: Path, file_name: str):
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
    if settings.workflow.resume:
        return

    for path in (checkpoint_path, checkpoint_path.with_name(f"{checkpoint_path.name}-shm"), checkpoint_path.with_name(f"{checkpoint_path.name}-wal")):
        path.unlink(missing_ok=True)


async def _generate_worker(plan: WorkerGenerationPlan, run_directory: Path) -> Path:
    worker_name = f"worker-{plan.worker_id:02d}"
    target_distribution = {
        class_name: bucket.target
        for class_name, bucket in plan.initial_state.input_distribution.items()
    }

    with logger.contextualize(worker=worker_name):
        logger.info(
            "Generating {} records from source classes: {}",
            plan.initial_state.target_size,
            list(plan.class_labels),
        )
        logger.info("Class distribution target: {}", target_distribution)

        output_file_name = (
            f"worker-{plan.worker_id:02d}.{settings.output_dataset.format}"
        )
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

    return run_directory / output_file_name


def _merge_worker_outputs(worker_files: list[Path]) -> None:
    output_directory = _output_directory()
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / _dataset_file_name()
    temporary_path = output_directory / (
        f".{final_path.stem}.{uuid4().hex}.tmp{final_path.suffix}"
    )

    try:
        merge_dataset_files(worker_files, temporary_path)
        os.replace(temporary_path, final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


async def generate_datasets_with_workers() -> None:
    plans = build_worker_generation_plans()
    output_directory = _output_directory()
    run_directory = output_directory / ".runs"
    if not settings.workflow.resume:
        shutil.rmtree(run_directory, ignore_errors=True)
        shutil.rmtree(output_directory / ".checkpoints", ignore_errors=True)
    run_directory.mkdir(parents=True, exist_ok=True)

    with inference_environment():
        async with asyncio.TaskGroup() as task_group:
            tasks = [
                task_group.create_task(_generate_worker(plan, run_directory))
                for plan in plans
            ]

    _merge_worker_outputs([task.result() for task in tasks])
    logger.info("Merged {} worker outputs into {}", len(plans), _dataset_file_name())
