# Working on this repo

- The current job is docs/rebuttal_plan.md. Work on one task at a time, only the one I name.
- No GPU and no checkpoints on this machine. Never load a real model or launch a real
  experiment. Use the synthetic adapters in tests/ and `python run.py --config <f> --dry-run`.
- Tests are standalone scripts: run `python tests/test_<name>.py`, never pytest.
  New tests follow the same style (print N/N checks passed, exit non-zero on failure).
- Additive only: never change existing strategies, adapters, operators or configs.
- New strategies follow docs/extending.md, all seven steps.
- Config comments must never say hyperparameters chosen on one checkpoint were reused on another.
- Commit at the end of each task with a message naming the task.