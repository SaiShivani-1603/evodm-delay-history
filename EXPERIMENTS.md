# Running the experiments

All commands run from the repository root in the conda environment `antibio` (Python 3.9, TensorFlow 2.11;
exact versions in `environment.yml` and `requirements-frozen.txt`).

1. Install: `python -m pip install -e .`
2. Tests: `python -m pytest evodm/test -q --deselect evodm/test/test_evol_game.py::test_init_dense`
3. Write the configuration files (already committed): `python scripts/make_configs.py`
4. Run one experiment: `python run_experiment.py configs/main/main_history_d2_s10.json`
5. Run all of them, 4 in parallel (finished runs are skipped): `caffeinate -i scripts/run_all.sh`

Each run writes to `results/raw/`: `<id>.csv` (per-episode fitness during training), `<id>_eval.csv`
(greedy evaluation on fresh episodes) and `<id>_record.json` (configuration, git commit, software versions,
hardware, compute time). Logs go to `logs/`.

Modes: `repo` = the repository's own agent (Conv1D, true current state); `state` = stale genotype only;
`history` = stale genotype plus the drugs given since; `decoy` = stale genotype plus random drugs
(same input size as `history`). Delay and seed are in the configuration file.
