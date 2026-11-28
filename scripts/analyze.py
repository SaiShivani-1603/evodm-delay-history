import glob, json, re, sys
import numpy as np, pandas as pd
from scipy import stats

RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw"
rows = []
for f in glob.glob(f"{RAW}/main_*_eval.csv"):
    m = re.search(r"main_(\w+?)_d(\d)_s(\d+)_eval", f)
    mode, d, s = m.group(1), int(m.group(2)), int(m.group(3))
    ev = pd.read_csv(f, keep_default_na=False)
    for (pol, ts), g in ev.groupby(["policy", "test_stale"]):
        rows.append((mode, d, s, pol, str(ts), g.fitness.mean()))
R = pd.DataFrame(rows, columns=["mode", "delay", "seed", "policy", "test", "fit"])
R.to_csv("/home/claude/work/analysis/per_run_eval.csv", index=False)

def rl(mode, d, test=None, seeds=None):
    t = str(d) if test is None else str(test)
    if mode == "repo": t = "train"
    x = R[(R["mode"] == mode) & (R.delay == d) & (R.policy == "RL") & (R.test == t)].set_index("seed").fit
    return x if seeds is None else x.loc[[s for s in seeds if s in x.index]]

def ci(diff):
    n = len(diff); m = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return m, m - t * se, m + t * se, n

def paired(a, b):
    idx = a.index.intersection(b.index); return ci(a.loc[idx] - b.loc[idx])

def fmt(x): return "%+.3f (95%% CI %+.3f to %+.3f, n=%d)" % x

print("=== Reference policies (greedy evaluation protocol, mean over runs) ===")
ref = R[R.policy.isin(["naive", "DP"])].groupby("policy").fit.agg(["mean", "std", "count"])
print(ref.round(3))
fc = pd.read_csv(f"{RAW}/fixed_cycles.csv"); best = fc.iloc[0]
print("best fixed cycle:", best.cycle, round(best.mean_fitness, 3), "| median 2-drug cycle", round(fc[fc.n_drugs == 2].mean_fitness.median(), 3))

print("\n=== Matched evaluation: mean fitness over seeds (lower is better), mean +- 95% CI half-width ===")
tab = []
for mode, d in [("repo", 0), ("state", 0)] + [(m, d) for d in (1, 2, 3) for m in ("state", "history", "decoy")]:
    x = rl(mode, d); m_, lo, hi, n = ci(x)
    tab.append((mode, d, n, round(m_, 3), round((hi - lo) / 2, 3), round(x.min(), 3), round(x.max(), 3)))
print(pd.DataFrame(tab, columns=["mode", "delay", "n", "mean", "ci_half", "min", "max"]).to_string(index=False))

print("\n=== H1: history - state (threshold <= -0.15, CI must exclude 0) ===")
for d in (1, 2, 3):
    print("delay", d, fmt(paired(rl("history", d), rl("state", d))))
print("\n=== H2: history - decoy (threshold <= -0.10) ===")
for d in (1, 2, 3):
    print("delay", d, fmt(paired(rl("history", d), rl("decoy", d))))
print("\n=== H3: trained at delay 2, tested at other staleness: history - state (threshold <= -0.10) ===")
for k in range(5):
    print("test staleness", k, fmt(paired(rl("history", 2, k), rl("state", 2, k))))
print("\n=== Fitness by test staleness for agents trained at each delay (mean over seeds) ===")
for mode in ("state", "history", "decoy"):
    for d in (1, 2, 3):
        print(mode, "trained d%d:" % d, [round(rl(mode, d, k).mean(), 3) for k in range(5)])
print("\n=== Agent vs best fixed cycle and vs random (paired not applicable; difference of means with seed CI) ===")
naive = ref.loc["naive", "mean"]
for mode, d in [("repo", 0), ("state", 0)] + [(m, d) for d in (1, 2, 3) for m in ("state", "history", "decoy")]:
    x = rl(mode, d); m_, lo, hi, n = ci(x - best.mean_fitness)
    print(f"{mode:8s} d{d}: vs best fixed cycle {m_:+.3f} [{lo:+.3f},{hi:+.3f}]  retained={(naive - x.mean())/(naive - ref.loc['DP','mean']):.2f}")
print("\n=== Reproduction check at matched budget: repo Conv1D vs Dense, delay 0 ===")
print("repo - Dense", fmt(paired(rl("repo", 0), rl("state", 0))))
