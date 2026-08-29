# Future Distribution Split

The project deliberately uses one Python distribution and a shared `src/` directory:

```text
src/
├── dataset_agent/
└── training/
```

This keeps development, testing, and dependency management simple while the pipeline is evolving.

## Split When Needed

Split into separate distributions only when one of these becomes true:

- `dataset_agent` needs an independent release cycle or external consumers.
- Training requires incompatible or substantially heavier dependencies.
- Separate teams own generation and training.
- Training runs need a separately deployable command-line interface.

## Migration Plan

1. Define stable boundaries before moving code: generated dataset schema, feature-artifact schema, model-artifact schema, and version metadata.
2. Create `packages/dataset-agent/` and move `src/dataset_agent/` to `packages/dataset-agent/src/dataset_agent/`.
3. Create `packages/training/` and move `src/training/` to `packages/training/src/training/`.
4. Add a `pyproject.toml` to each package. Keep package-specific runtime dependencies there.
5. Add a root workspace declaration:

```toml
[tool.uv.workspace]
members = ["packages/dataset-agent", "packages/training"]
```

6. In `training/pyproject.toml`, depend on the generation package through the workspace:

```toml
[project]
dependencies = ["dataset-agent"]

[tool.uv.sources]
dataset-agent = { workspace = true }
```

7. Replace cross-package internal imports with public APIs. Training should load generated artifacts rather than importing generation internals.
8. Run `uv lock` from the repository root to create one workspace `uv.lock`, then run each package's tests and CLI entry points.
9. Publish or deploy packages independently only after their data contracts and versioning are stable.
