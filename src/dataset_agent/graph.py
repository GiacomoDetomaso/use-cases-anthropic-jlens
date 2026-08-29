from pathlib import Path

from langgraph.graph import END, START, StateGraph

from app.settings import settings
from dataset_agent.core.writers.writers_builder import (
    build_dataset_writer,
)
from dataset_agent.dataset_source import (
    get_source_dataset,
    get_target_transformation_examples_dataset,
)
from dataset_agent.models.dataset_generation_state_model import (
    DatasetState,
    DistributionState,
)
from dataset_agent.nodes.discard import discard_node
from dataset_agent.nodes.generator import (
    generation_router,
    generator_node_async,
    generator_node_sync,
)
from dataset_agent.nodes.pickers import (
    PickerInputDatasetNode,
    PickerTargetDatasetNode,
)
from dataset_agent.nodes.repair import repair_node_async, repair_node_sync
from dataset_agent.nodes.save import FlushNode, SaveNode, save_router
from dataset_agent.nodes.validator import (
    validation_router,
    validator_node_async,
    validator_node_sync,
)

source_dataset = get_source_dataset()
target_dataset = get_target_transformation_examples_dataset()


def _use_async_nodes() -> bool:
    vllm_settings = settings.workflow.inference.vllm
    return settings.workflow.workers > 1 or (
        vllm_settings is not None and vllm_settings.invoke_mode == "async"
    )


def build_and_compile_graph(
    target_size: int | None = None,
    output_path: Path | None = None,
    output_file_name: str = "dataset.csv",
    seed: int = 42,
    checkpointer=None,
):
    target_size = target_size or settings.output_dataset.target_size
    output_path = output_path or Path(__file__).resolve().parents[2] / "output"
    input_picker = PickerInputDatasetNode(
        dataset=source_dataset,
        dataset_settings=settings.source_dataset,
        seed=seed,
    )

    target_picker = PickerTargetDatasetNode(
        dataset=target_dataset,
        dataset_settings=settings.target_transformation_examples_dataset,
        target_size=target_size,
        examples_count=1,
        seed=seed,
    )

    save = SaveNode(
        writer=build_dataset_writer(
            output_path=output_path,
            file_name=output_file_name,
        )
    )
    flush = FlushNode(writer=save.writer)

    def to_graph_node(node):
        def wrapped(state):
            if isinstance(state, dict):
                state = DatasetState(**state)

            updated_state = node(state)
            return updated_state.model_dump()

        return wrapped

    builder = StateGraph(DatasetState)

    builder.add_node("pick_input", to_graph_node(input_picker))
    builder.add_node("pick_target", to_graph_node(target_picker))
    generator = generator_node_async if _use_async_nodes() else generator_node_sync
    validator = validator_node_async if _use_async_nodes() else validator_node_sync
    repair = repair_node_async if _use_async_nodes() else repair_node_sync
    builder.add_node("generator", generator)
    builder.add_node("validator", validator)
    builder.add_node("repair", repair)
    builder.add_node("discard", discard_node)
    builder.add_node("save", save)
    builder.add_node("flush", flush)

    builder.add_edge(START, "pick_input")
    builder.add_edge("pick_input", "pick_target")
    builder.add_edge("pick_target", "generator")

    builder.add_conditional_edges(
        "generator",
        generation_router,
        {
            "success": "validator",
            "retry": "pick_input",
            "completed": "flush",
        },
    )

    builder.add_conditional_edges(
        "validator",
        validation_router,
        {
            "accepted": "save",
            "repair": "repair",
            "discard": "discard",
        },
    )

    builder.add_edge("repair", "validator")

    builder.add_conditional_edges(
        "discard",
        save_router,
        {
            "completed": "flush",
            "not_completed": "pick_input",
        },
    )

    builder.add_conditional_edges(
        "save", save_router, {"completed": "flush", "not_completed": "pick_input"}
    )

    builder.add_edge("flush", END)

    return builder.compile(checkpointer=checkpointer)


def get_graph_initial_state(
    target_size: int | None = None,
    remaining_input_indices: list[int] | None = None,
    input_distribution: DistributionState | None = None,
):
    return DatasetState(
        target_size=target_size or settings.output_dataset.target_size,
        index_to_generate=0,
        remaining_input_indices=(
            remaining_input_indices
            if remaining_input_indices is not None
            else source_dataset.index.tolist()
        ),
        input_distribution=input_distribution or {},
    )
