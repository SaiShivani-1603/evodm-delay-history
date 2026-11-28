"""Run one experiment from a configuration file.

usage: python run_experiment.py configs/main/main_history_d2_s10.json [--force]

Config keys: id, mode (repo | state | history | decoy), delay, seed, episodes, train_freq,
eval_episodes, eval_stale (list; ignored for mode repo), network (dense | conv; optional,
default dense; delayed-observation modes only).

Writes, under results/raw/:
  <id>.csv          per-episode mean fitness during training (columns episode, RL, naive, DP)
  <id>_eval.csv     greedy evaluation on fresh episodes (policy, test_stale, episode, fitness)
  <id>_trace.csv    every evaluation step: drug given and true genotype (for the drug-use analysis)
  <id>_dp_policy.npy  the exact optimal policy for this landscape, for comparison with the trace
  <id>_record.json  config, git commit, software versions, hardware, compute time, summary
A run whose record already exists is skipped unless --force is given.
"""
import json, os, platform, random, subprocess, sys, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf

from evodm.exp import evol_deepmind
from evodm.evaluate import per_episode, greedy_eval, reference_eval

MODES = ("repo", "state", "history", "decoy")


def git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main(path, force=False):
    cfg = json.load(open(path))
    rid, mode = cfg["id"], cfg["mode"]
    if mode not in MODES:
        sys.exit("mode must be one of %s" % (MODES,))
    out = "results/raw"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/{rid}_record.json") and not force:
        print("already done:", rid)
        return

    seed = int(cfg["seed"])
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    delay = int(cfg.get("delay", 0))
    kw = dict(episodes=int(cfg["episodes"]), N=int(cfg.get("N", 4)),
              num_drugs=int(cfg.get("num_drugs", 15)), mira=True,
              train_freq=int(cfg["train_freq"]))
    if mode == "repo":
        kw.update(train_input="state_vector", delay=delay)
    else:
        kw.update(train_input="delayed_" + mode, obs_delay=delay, delay=0,
                  network=cfg.get("network", "dense"))

    start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    res = evol_deepmind(**kw)
    agent, naive_agent, dp_agent = res[2], res[3], res[4]
    train = {"RL": per_episode(agent), "naive": per_episode(naive_agent), "DP": per_episode(dp_agent)}
    n = min(len(v) for v in train.values())
    df = pd.DataFrame({k: v[:n] for k, v in train.items()})
    df.index.name = "episode"
    df.to_csv(f"{out}/{rid}.csv")
    t_train = time.time() - t0

    # greedy evaluation on fresh episodes (no exploration, no learning)
    t1 = time.time()
    n_eval = int(cfg.get("eval_episodes", 100))
    rows, trace_rows = [], []
    stales = [None] if mode == "repo" else list(cfg.get("eval_stale", [delay]))
    for s in stales:
        steps = []
        fit = greedy_eval(agent, stale=s, episodes=n_eval, trace=steps)
        trace_rows += [("train" if s is None else s, *r) for r in steps]
        rows += [("RL", "train" if s is None else s, i, float(f)) for i, f in enumerate(fit)]
    for name, ag, kind in (("naive", naive_agent, "naive"), ("DP", dp_agent, "dp")):
        fit = reference_eval(ag, kind, episodes=n_eval)
        rows += [(name, "NA", i, float(f)) for i, f in enumerate(fit)]
    ev = pd.DataFrame(rows, columns=["policy", "test_stale", "episode", "fitness"])
    ev.to_csv(f"{out}/{rid}_eval.csv", index=False)
    pd.DataFrame(trace_rows, columns=["test_stale", "episode", "action_number", "action", "genotype",
                                      "fitness"]).to_csv(f"{out}/{rid}_trace.csv", index=False)
    np.save(f"{out}/{rid}_dp_policy.npy", np.asarray(res[6]))
    t_eval = time.time() - t1

    summary = {f"{p}_stale{s}": round(float(g.fitness.mean()), 4)
               for (p, s), g in ev.groupby(["policy", "test_stale"])}
    summary["train_last50_RL"] = round(float(df.RL.values[-50:].mean()), 4)
    record = {
        "id": rid, "config": cfg, "started_utc": start,
        "seconds_train": round(t_train, 1), "seconds_eval": round(t_eval, 1),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(), "tensorflow": tf.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(), "summary": summary,
    }
    json.dump(record, open(f"{out}/{rid}_record.json", "w"), indent=2)
    print(rid, summary, "train s:", record["seconds_train"], "eval s:", record["seconds_eval"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], force="--force" in sys.argv)
