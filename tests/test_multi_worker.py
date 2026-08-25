import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dataset_agent import graph_invoker
from dataset_agent.worker_generation_plan import build_worker_generation_plans
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
            first_worker_file = output_directory / "worker-01.jsonl"
            second_worker_file = output_directory / "worker-02.jsonl"
            first_worker_file.write_text("first\n", encoding="utf-8")
            second_worker_file.write_text("second\n", encoding="utf-8")

            with patch.object(
                graph_invoker, "_output_directory", return_value=output_directory
            ):
                graph_invoker._merge_worker_outputs([first_worker_file, second_worker_file])

            self.assertEqual(
                (output_directory / "dataset.jsonl").read_text(encoding="utf-8"),
                "first\nsecond\n",
            )


if __name__ == "__main__":
    unittest.main()