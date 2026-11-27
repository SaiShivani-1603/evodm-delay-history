# Changes relative to the upstream code

Upstream: https://github.com/DavisWeaver/evo_dm, commit `d975114` (GPL-3.0). This fork keeps the
upstream licence file and is released under GPL-3.0. Everything not listed here is unchanged upstream code.

## Modified upstream files
- `evodm/evol_game.py`: new `obs_delay` argument and three `train_input` modes (`delayed_state`,
  `delayed_history`, `delayed_decoy`); histories of states and drugs; `build_obs()`; `OBS_STALE`
  (real staleness of the reading, separate from `OBS_DELAY`, the window length the network was built for);
  new input shapes; replay tuples built from the delayed observation in `define_sensor`.
- `evodm/learner.py`: `OBS_DELAY` hyperparameter passed to the environment; the Dense network
  is used for the new modes; `get_qs` reads the current observation for the new modes.
- `evodm/exp.py`: `obs_delay` argument of `evol_deepmind`.

## New files (mine)
- `evodm/evaluate.py`: greedy evaluation of a trained agent (no exploration, no learning), with
  optional mismatched staleness, and evaluation of the random and exact-optimum reference policies.
- `run_experiment.py`: runs one experiment from a configuration file and writes the raw results and a run record.
- `run_baseline.py`: earlier command-line runner, used for the pilot runs (results `baseline_*`, `state_*`,
  `history_*`, `decoy_*` with `_ep200_` or `_ep500_` in their names); kept for traceability.
- `scripts/make_configs.py`, `scripts/run_all.sh`: build and run every configuration.
- `configs/main/`: one configuration file per experiment in the paper.
- `evodm/test/test_delay_obs.py`, `evodm/test/test_mismatch_eval.py`: tests for the new code.
- `REPRODUCTION.md`, `DESIGN_FREEZE.md`, `EXPERIMENTS.md`, `CHANGES.md`: documentation.

## Not changed
The evolutionary simulator, the landscapes, the Q-learning update, the replay and exploration logic, the
exact optimum (backward induction) and the random policy.
