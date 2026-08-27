import asyncio
import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from dataset_agent import graph_invoker
from dataset_agent.core.writers.writers import (
    CsvDatasetWriter,
    JsonLDatasetWriter,
    SyntheticRecord,
)
from dataset_agent.core.writers.writers_builder import (
    merge_dataset_files,
)
from dataset_agent.graph import build_and_compile_graph
from dataset_agent.worker_generation_plan import build_worker_generation_plans
import main as application_main
from settings import Settings, settings


class MultiWorkerGenerationTests(unittest.TestCase):
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
            label: 100.0 if index == 0 else 0.0
            for index, label in enumerate(labels)
        }

        with self.assertRaisesRegex(ValueError, "balanced.*random"):
            Settings.model_validate(config)

    def test_worker_outputs_merge_in_worker_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            first_worker_file = output_directory / "worker-01.csv"
            second_worker_file = output_directory / "worker-02.csv"
            first_worker_file.write_text("source,target,output\nfirst,first,first\n", encoding="utf-8")
            second_worker_file.write_text("source,target,output\nsecond,second,second\n", encoding="utf-8")

            with patch.object(
                graph_invoker, "_output_directory", return_value=output_directory
            ):
                graph_invoker._merge_worker_outputs([first_worker_file, second_worker_file])

            self.assertEqual(
                list(csv.DictReader((output_directory / "dataset.csv").open(encoding="utf-8"))),
                [
                    {"source": "first", "target": "first", "output": "first"},
                    {"source": "second", "target": "second", "output": "second"},
                ],
            )

    def test_csv_worker_shard_restores_records_without_duplicates(self):
        record = SyntheticRecord(
            source={"intent": "source"},
            target={"category": "target"},
            output="output",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            writer = CsvDatasetWriter(output_directory, "worker-01.csv")
            self.assertTrue(writer.append_at(0, record))
            writer.serialize(start=0, stop=1)

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
            first_writer.append(SyntheticRecord("first", "first", "first"))
            second_writer.append(SyntheticRecord("second", "second", "second"))
            first_writer.serialize(start=0, stop=1)
            second_writer.serialize(start=0, stop=1)

            destination = output_directory / "dataset.jsonl"
            merge_dataset_files([first_worker_file, second_worker_file], destination)

            self.assertEqual(
                JsonLDatasetWriter(output_directory, destination.name).dataset,
                [
                    SyntheticRecord("first", "first", "first"),
                    SyntheticRecord("second", "second", "second"),
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

        record = SyntheticRecord(
            source={"intent": "source"},
            target={"category": "target"},
            output="output",
        )
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

    def test_graph_uses_supplied_checkpointer(self):
        checkpointer = InMemorySaver()
        graph = build_and_compile_graph(checkpointer=checkpointer)

        self.assertIs(graph.checkpointer, checkpointer)

    def test_async_sqlite_checkpointer_reads_empty_worker_state(self):
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
                    snapshot = await graph.aget_state(
                        {"configurable": {"thread_id": "worker-01"}}
                    )
                    self.assertFalse(snapshot.values)

        asyncio.run(check())

    def test_async_main_awaits_multi_worker_generation(self):
        with (
            patch.object(application_main, "setup_logger"),
            patch.object(
                graph_invoker,
                "generate_datasets_with_workers",
                new_callable=AsyncMock,
            ) as generate_workers,
        ):
            asyncio.run(application_main.main())

        generate_workers.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()