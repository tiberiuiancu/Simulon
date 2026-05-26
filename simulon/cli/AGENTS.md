# cli: Command-Line Interface

957 lines total across 3 files. Typer-based CLI, the primary user-facing entry point.

## OVERVIEW

Two command groups: `simulate` (run simulation) and `install` (third-party deps). Also `trace generate` for trace-driven DAG extraction.

## STRUCTURE

```
cli/
├── __init__.py   # Main CLI: app definition, simulate command (~308 lines)
├── trace.py      # Trace generation: `simulon trace generate` (~367 lines)
└── install.py    # `simulon install apex` / `simulon install deepgemm` (~282 lines)
```

## WHERE TO LOOK

| Task | Location in `__init__.py` | Notes |
|------|---------------------------|-------|
| Modify simulate output | Lines 102-225 | `simulate()` function, Chrome trace, GOAL, energy, cost, verbose |
| Add simulate option | Lines 102-116 | Typer Option() annotations |
| Add new CLI subcommand | Top of file | `app.command()` or new Typer subgroup |
| Trace generation | `trace.py` | `simulon trace generate` command |
| Add installable package | `install.py` | Follow `install_apex` pattern |

## CONVENTIONS

- **Typer framework**, `app = typer.Typer()` with `@app.command()` decorators
- **Rich console**, `from rich.console import Console` for formatted output
- **`_print_*` helpers**, `_print_summary`, `_print_energy_summary`, `_print_cost_summary` for display
- **Energy + cost**, optional `--energy` and `--cost` flags in `simulate` compute per-iteration power and cost models

## ANTI-PATTERNS

- **`__init__.py` is ~308 lines** but contains formatting helpers `_print_summary`, `_print_energy_summary`, and `_print_cost_summary` inline. Consider splitting display logic into a `display.py` if adding more output modes
