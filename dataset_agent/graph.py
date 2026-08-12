from langgraph.graph import StateGraph, START, END

from dataset_agent.dataset_source import (
    get_source_dataset,
    get_target_transformation_examples_dataset,
)
from dataset_agent.models.dataset_generation_state_model import DatasetState
from dataset_agent.nodes.pickers import PickerInputDatasetNode, PickerTargetDatasetNode
from dataset_agent.nodes.generator import (
    generation_router,
    generator_node_sync,
    generator_node_async,
)
from dataset_agent.nodes.save import SaveNode, save_router
from dataset_agent.core.writers import JsonLDatasetWriter
from settings import settings

from pathlib import Path

source_dataset = get_source_dataset()
target_dataset = get_target_transformation_examples_dataset()

def build_and_compile_graph():
    input_picker = PickerInputDatasetNode(
        dataset=source_dataset,
        dataset_settings=settings.source_dataset,
        seed=42,
    )

    target_picker = PickerTargetDatasetNode(
        dataset=target_dataset,
        dataset_settings=settings.target_transformation_examples_dataset,
        target_size=settings.output_dataset.target_size,
        examples_count=3,
        seed=42,
    )

    save = SaveNode(
        writer=JsonLDatasetWriter(
            output_path=Path(Path(__file__).resolve().parent.parent / "output")
        )
    )

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
    vllm_settings = settings.workflow.inference.vllm
    generator = (
        generator_node_async
        if vllm_settings is not None and vllm_settings.invoke_mode == "async"
        else generator_node_sync
    )
    builder.add_node("generator", generator)
    builder.add_node("save", save)

    builder.add_edge(START, "pick_input")
    builder.add_edge("pick_input", "pick_target")
    builder.add_edge("pick_target", "generator")

    builder.add_conditional_edges(
        "generator",
        generation_router,
        {
            "success": "save",      # later validator
            "retry": "pick_input",
            "completed": END,
        },
    )

    builder.add_conditional_edges(
        "save",
        save_router,
        {
            "completed": END,
            "not_completed": "pick_input"
        }
    )

    return builder.compile()

def get_graph_initial_state():
    return DatasetState(
        target_size=settings.output_dataset.target_size,
        index_to_generate=0,
        remaining_input_indices=source_dataset.index.tolist(),
    )
