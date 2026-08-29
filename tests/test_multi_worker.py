import asyncio
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app import main as application_main
from app.settings import Settings, settings
from dataset_agent import graph_invoker
from dataset_agent.core.writers.writers import (
    CsvDatasetWriter,
    JsonLDatasetWriter,
    SyntheticRecord,
    _csv_fieldnames,
)
from dataset_agent.core.writers.writers_builder import (
    merge_dataset_files,
)
from dataset_agent.graph import build_and_compile_graph
from dataset_agent.models.dataset_generation_io_models import (
    InputAttackModel,
    InputDatasetModel,
    OutputModel,
)
from dataset_agent.worker_generation_plan import (
    build_worker_generation_plans,
)


class MultiWorkerGenerationTests(unittest.TestCase):
    def test_csv_schema_rejects_duplicate_model_field_names(self):
        class FirstModel(BaseModel):
            shared: str

        class SecondModel(BaseModel):
            shared: str

        with self.assertRaisesRegex(KeyError, "Duplicate CSV field names: shared"):
            _csv_fieldnames({"first": FirstModel, "second": SecondModel})

    def test_worker_plans_partition_classes_and_preserve_total_quota(self):
        workflow = settings.workflow
        original_workers = workflow.workers
        original_groups = workflow.worker_class_groups
        labels = settings.source_dataset.class_labels

        try:
            workflow.workers = 2
            workflow.worker_class_groups = [labels[:13], labels[13:]]

            plans = build_worker_generation_plans()
        finally:
            workflow.workers = original_workers
            workflow.worker_class_groups = original_groups

        assigned_labels = [label for plan in plans for label in plan.class_labels]
        self.assertEqual(len(plans), 2)
        self.assertEqual(set(assigned_labels), set(labels))
        self.assertEqual(len(assigned_labels), len(set(assigned_labels)))
        self.assertEqual(
            sum(plan.initial_state.target_size for plan in plans),
            settings.output_dataset.target_size,
        )

    def test_multi_worker_rejects_manual_source_distribution(self):
        config = settings.model_dump()
        labels = settings.source_dataset.class_labels
        config["workflow"]["workers"] = 2
        config["workflow"]["worker_class_groups"] = [labels[:13], labels[13:]]
        config["source_dataset"]["class_distribution"] = {
            label: 100.0 if index == 0 else 0.0 for index, label in enumerate(labels)
        }

        with self.assertRaisesRegex(ValueError, "balanced.*random"):
            Settings.model_validate(config)

    def test_worker_outputs_merge_in_worker_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            first_worker_file = output_directory / "worker-01.csv"
            second_worker_file = output_directory / "worker-02.csv"
            first_writer = CsvDatasetWriter(output_directory, first_worker_file.name)
            second_writer = CsvDatasetWriter(output_directory, second_worker_file.name)
            first_writer.append(self._record("first"))
            second_writer.append(self._record("second"))
            first_writer.serialize(start=0, stop=1)
            second_writer.serialize(start=0, stop=1)

            with patch.object(
                graph_invoker, "_output_directory_path", return_value=output_directory
            ):
                graph_invoker._merge_worker_outputs([first_worker_file, second_worker_file])

            dataset_file_name = f"{settings.output_dataset.name}.{settings.output_dataset.format}"
            self.assertEqual(
                CsvDatasetWriter(output_directory, dataset_file_name).dataset,
                [self._record("first"), self._record("second")],
            )

    def test_csv_worker_shard_restores_records_without_duplicates(self):
        record = self._record("first")

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            writer = CsvDatasetWriter(output_directory, "worker-01.csv")
            self.assertTrue(writer.append_at(0, record))
            writer.serialize(start=0, stop=1)

            with (output_directory / "worker-01.csv").open(encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(
                set(row),
                {
                    "target_intent",
                    "target_examples",
                    "target_description",
                    "original_prompt",
                    "original_intent",
                    "text",
                },
            )
            self.assertEqual(row["text"], "first output")

            resumed_writer = CsvDatasetWriter(output_directory, "worker-01.csv")
            self.assertEqual(resumed_writer.dataset, [record])
            self.assertFalse(resumed_writer.append_at(0, record))

    def test_jsonl_worker_shards_merge_with_the_writer_api(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            first_worker_file = output_directory / "worker-01.jsonl"
            second_worker_file = output_directory / "worker-02.jsonl"
            first_writer = JsonLDatasetWriter(output_directory, first_worker_file.name)
            second_writer = JsonLDatasetWriter(output_directory, second_worker_file.name)
            first_writer.append(self._record("first"))
            second_writer.append(self._record("second"))
            first_writer.serialize(start=0, stop=1)
            second_writer.serialize(start=0, stop=1)

            destination = output_directory / "dataset.jsonl"
            merge_dataset_files([first_worker_file, second_worker_file], destination)

            self.assertEqual(
                JsonLDatasetWriter(output_directory, destination.name).dataset,
                [
                    self._record("first"),
                    self._record("second"),
                ],
            )

    def test_resume_reactivates_the_serialized_checkpoint(self):
        class Graph:
            def __init__(self, snapshot):
                self.snapshot = snapshot
                self.update_state_calls = []

            def get_state_history(self, config):
                return iter([self.snapshot])

            def update_state(self, config, values, as_node):
                self.update_state_calls.append((config, values, as_node))
                return {"configurable": {"checkpoint_id": "resumed"}}

        record = self._record("first")
        snapshot = SimpleNamespace(
            config={"configurable": {"checkpoint_id": "saved"}},
            values={
                "index_to_generate": 1,
                "last_checkpoint_index": 1,
                "source": None,
                "target": None,
                "generated_prompt": None,
                "regenerated_prompt": None,
                "validation_output": None,
            },
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            writer = CsvDatasetWriter(output_directory, "worker-01.csv")
            writer.append(record)
            writer.serialize(start=0, stop=1)
            graph = Graph(snapshot)

            config = graph_invoker._restore_resumable_sync_state(
                graph,
                {"configurable": {"thread_id": "worker-01"}},
                output_directory,
                "worker-01.csv",
            )

        self.assertEqual(config, {"configurable": {"checkpoint_id": "resumed"}})
        self.assertEqual(
            graph.update_state_calls,
            [(snapshot.config, {}, "save")],
        )

    @staticmethod
    def _record(value: str) -> SyntheticRecord:
        return SyntheticRecord(
            source=InputDatasetModel(
                original_prompt=f"{value} prompt",
                original_intent=f"{value} intent",
            ),
            target=InputAttackModel(
                target_intent=settings.target_transformation_examples_dataset.class_labels[0],
                target_examples=f"{value} examples",
            ),
            output=OutputModel(text=f"{value} output"),
        )

    def test_graph_uses_supplied_checkpointer(self):
        checkpointer = InMemorySaver()
        graph = build_and_compile_graph(checkpointer=checkpointer)

        self.assertIs(graph.checkpointer, checkpointer)

    def test_async_sqlite_checkpointer_reads_empty_worker_state(self) -> None:
        async def check() -> None:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            with tempfile.TemporaryDirectory() as temporary_directory:
                output_directory = Path(temporary_directory)
                async with AsyncSqliteSaver.from_conn_string(
                    str(output_directory / "state.sqlite")
                ) as checkpointer:
                    graph = build_and_compile_graph(
                        output_path=output_directory,
                        checkpointer=checkpointer,
                    )
                    snapshot = await graph.aget_state({"configurable": {"thread_id": "worker-01"}})
                    self.assertFalse(snapshot.values)

        asyncio.run(check())

    def test_async_main_awaits_multi_worker_generation(self) -> None:
        with (
            patch.object(application_main, "setup_logger"),
            patch.object(
                graph_invoker,
                "generate_datasets_with_workers",
                new_callable=AsyncMock,
            ) as generate_workers,
        ):
            asyncio.run(application_main.run())

        generate_workers.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
