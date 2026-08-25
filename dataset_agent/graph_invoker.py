from operator import imod
import os
import shutil
import asyncio

from loguru import logger
from pathlib import Path
from uuid import uuid4

from dataset_agent.graph import build_and_compile_graph, get_graph_initial_state
from dataset_agent.core.inference.inference_context_manager import inference_environment
from dataset_agent.worker_generation_plan import (
    WorkerGenerationPlan,
    build_worker_generation_plans,
)

def generate_dataset() -> None:
    """
    Primitive to run the graph in synchronous mode
    """
    initial_state = get_graph_initial_state()
    graph = build_and_compile_graph()

    # Wrap graph execution inside the environment
    with inference_environment():
        graph.invoke(initial_state)


async def generate_dataset_async() -> None:
    """
    Primitive to run the graph in asynchronous mode
    """
    initial_state = get_graph_initial_state()
    graph = build_and_compile_graph()
    
    # Wrap graph execution inside the environment
    with inference_environment():
        await graph.ainvoke(initial_state)

def _output_directory() -> Path:
    return Path(__file__).resolve().parent / "output"


async def _generate_worker(plan: WorkerGenerationPlan, run_directory: Path) -> Path:
    logger.info(
        "Worker {} generating {} records from source classes: {}",
        plan.worker_id,
        plan.initial_state.target_size,
        list(plan.class_labels),
    )

    output_file_name = f"worker-{plan.worker_id:02d}.jsonl"

    graph = build_and_compile_graph(
        target_size=plan.initial_state.target_size,
        output_path=run_directory,
        output_file_name=output_file_name,
        seed=42 + plan.worker_id,
    )

    await graph.ainvoke(plan.initial_state)
    
    return run_directory / output_file_name


def _merge_worker_outputs(worker_files: list[Path]) -> None:
    output_directory = _output_directory()
    output_directory.mkdir(parents=True, exist_ok=True)
    final_path = output_directory / "dataset.jsonl"
    temporary_path = output_directory / f".{final_path.name}.{uuid4().hex}.tmp"

    try:
        with open(temporary_path, "w", encoding="utf-8") as destination:
            for worker_file in worker_files:
                with open(worker_file, "r", encoding="utf-8") as source:
                    shutil.copyfileobj(source, destination)
        os.replace(temporary_path, final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


async def generate_datasets_with_workers() -> None:
    plans = build_worker_generation_plans()
    run_directory = _output_directory() / ".runs" / uuid4().hex
    run_directory.mkdir(parents=True, exist_ok=False)

    with inference_environment():
        async with asyncio.TaskGroup() as task_group:
            tasks = [
                task_group.create_task(_generate_worker(plan, run_directory))
                for plan in plans
            ]

    _merge_worker_outputs([task.result() for task in tasks])
    shutil.rmtree(run_directory)
    logger.info("Merged {} worker outputs into dataset.jsonl", len(plans))