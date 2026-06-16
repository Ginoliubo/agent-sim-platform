# Contributing to Agent Sim Platform

## Development Setup

```bash
# Clone the repository
git clone git@github.com:Ginoliubo/agent-sim-platform.git
cd agent-sim-platform

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
python3 -m pip install -e ".[dev]"
```

## Git Workflow

1. **Branch**: Create a feature branch from `main`.
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Commit**: Make focused commits with descriptive messages.
   ```bash
   git add <files>
   git commit -m "Add feature X"
   ```

3. **Test**: Run the full test suite before pushing.
   ```bash
   python3 -m pytest tests/ -q
   ```

4. **Push**: Push your branch and open a pull request.
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

This project uses `black` and `ruff`:

```bash
black agent_sim_platform/ tests/
ruff check agent_sim_platform/ tests/
```

Configuration is in `pyproject.toml`.

## Adding a Benchmark Fixture

1. Create a YAML file in `agent_sim_platform/benchmarks/fixtures/`.
2. Include `source` and `source_url` for traceability.
3. Document assumptions in `notes`.
4. Add a test in `tests/test_benchmarks.py`.

## Adding a New Algorithm Family

1. Add the preset to `agent_sim_platform/algorithms/families.py`.
2. Bind it to the relevant model preset in `agent_sim_platform/models/presets.py`.
3. Add tests in `tests/test_algorithms.py`.

## Release Checklist

- [ ] All tests pass: `pytest tests/`
- [ ] New commands are documented in `README.md`
- [ ] New concepts are documented in `docs/`
- [ ] CHANGELOG is updated (if applicable)
