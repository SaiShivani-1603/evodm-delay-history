import glob, re, sys
import numpy as np, pandas as pd
from scipy import stats
RAW = sys.argv[1] if len(sys.argv) > 1 else "results/raw"
def H(c):
    p = np.bincount(c, minlength=16)[1:16] / len(c); p = p[p > 0]; return -(p * np.log2(p)).sum()
rows = []
for f in sorted(glob.glob(f"{RAW}/main_*_trace.csv")):
    mode, d, s = re.search(r"main_(\w+?)_d(\d)_s(\d+)_trace", f).groups(); d, s = int(d), int(s)
    t = pd.read_csv(f, keep_default_na=False); dp = np.load(f.replace("_trace.csv", "_dp_policy.npy"))
    ts = "train" if mode == "repo" else str(d)
    t = t[(t.test_stale.astype(str) == ts) & (t.action_number >= 2)].sort_values(["episode", "action_number"])
    t["genotype"] = t.genotype.astype(int); t["action"] = t.action.astype(int)
    agree = np.mean([(a - 1) == dp[g, min(n - 1, dp.shape[1] - 1)] for g, a, n in zip(t.genotype, t.action, t.action_number)])
    prev = t.groupby("episode").action.shift(); sw = (t.action != prev)[prev.notna()].mean()
    top2 = np.sort(np.bincount(t.action, minlength=16)[1:])[::-1][:2].sum() / len(t)
    ep_ent = np.mean([H(g.action.values) for _, g in t.groupby("episode")])
    rows.append(dict(mode=mode, delay=d, seed=s, entropy_bits=H(t.action.values), within_ep_entropy=ep_ent,
                     distinct=t.action.nunique(), top2_share=top2, switch_rate=sw, dp_agree=agree,
                     frac_geno0=(t.genotype == 0).mean(), fit=t.fitness.mean()))
M = pd.DataFrame(rows); M.to_csv("results/mechanism_per_run.csv", index=False)
pd.set_option("display.width", 200)
g = M.groupby(["mode", "delay"])[["entropy_bits", "within_ep_entropy", "distinct", "top2_share", "switch_rate", "dp_agree", "frac_geno0", "fit"]].mean().round(3)
print(g.to_string())
print("\nCorrelation across runs (all agent runs): fitness vs dp_agree %.2f, vs top2_share %.2f, vs switch_rate %.2f" % tuple(
    stats.pearsonr(M.fit, M[c])[0] for c in ("dp_agree", "top2_share", "switch_rate")))
# learning curves: mean over seeds, 50-episode bins
print("\n=== Learning curves (training-time per-episode fitness, mean over seeds, 50-episode bins) ===")
out = {}
for (mode, d), _ in M.groupby(["mode", "delay"]):
    cs = []
    for f in glob.glob(f"{RAW}/main_{mode}_d{d}_s*.csv"):
        if re.search(r"_(eval|trace|record)", f): continue
        c = pd.read_csv(f); cs.append(c.RL.groupby(np.arange(len(c)) // 50).mean().values)
    out[f"{mode} d{d}"] = np.mean(cs, axis=0)
L = pd.DataFrame(out, index=[f"{i*50}-{i*50+49}" for i in range(10)]).round(3)
print(L.T.to_string()); L.to_csv("results/learning_curves_binned.csv")
