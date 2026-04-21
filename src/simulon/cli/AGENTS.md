# cli: Command-Line Interface

2 files, 755 lines total. Typer-based CLI, the primary user-facing entry point.

## OVERVIEW

Three command groups: `simulate` (run simulation), `profile` (gpu/node benchmarking), and `install` (third-party deps). Entry point: `simulon = "simulon.cli:app"` in pyproject.toml.

## STRUCTURE

```
cli/
├── __init__.py   # Main CLI: app definition, simulate command, profile gpu/node commands (755 lines)
└── install.py    # `simulon install apex` / `simulon install deepgemm` subcommands
```

## WHERE TO LOOK

| Task | Location in `__init__.py` | Notes |
|------|---------------------------|-------|
| Modify simulate output | Lines 36-149 | `simulate()` function, Chrome trace, GOAL, energy, cost, verbose |
| Add simulate option | Lines 36-80 | Typer Option() annotations |
| Modify GPU profiling | Lines 248-558 | `profile_gpu()`, sweep config, skip logic, OOM tracking |
| Modify node profiling | Lines 589-754 | `profile_node()`, nccl-tests JSON parsing |
| Add new subcommand | Top of file | `app.command()` or new Typer subgroup |
| Add installable package | `install.py` | Follow `install_apex` pattern |

## CONVENTIONS

- **Typer framework**, `app = typer.Typer()` with `@app.command()` decorators
- **Rich console**, `from rich.console import Console` for formatted output
- **`_print_*` helpers**, `_print_summary`, `_print_energy_summary`, `_print_cost_summary` for display
- **Incremental profiling**, `profile_gpu` appends to existing YAML templates, skipping already-profiled configs

## ANTI-PATTERNS

- **`__init__.py` is 755 lines**, all three major commands in one file. Consider splitting if adding more commands
- **Skip logic duplicated**, `_config_done` in CLI mirrors skip logic in `profiling/kernels.py`

