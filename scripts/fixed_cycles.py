"""Fixed drug cycles that need no readings (so measurement delay cannot hurt them).

Evaluates every ordered two-drug cycle (a, b, a, b, ...) and every single drug on the same simulator,
landscapes and 20-step episodes as the agents, and writes results/raw/fixed_cycles.csv.
Metric: mean fitness over the 20 chosen steps of each episode, averaged over episodes (lower is better),
the same as the agents' evaluation.
usage: python scripts/fixed_cycles.py [episodes_per_cycle] [seed]
"""
import os, random, sys
import numpy as np
import pandas as pd
from evodm import evol_env, define_mira_landscapes

episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 100
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
random.seed(seed); np.random.seed(seed)
N, K, STEPS = 4, 15, 20

env = evol_env(N=N, num_drugs=K, drugs=define_mira_landscapes(), train_input="state_vector",
               add_noise=False, normalize_drugs=True)  # same landscape settings as evol_deepmind(mira=True)

def run(cycle):
    per_episode = []
    for _ in range(episodes):
        env.reset()
        env.step()                       # first step uses the default drug, as in the agents' episodes
        fit = []
        for t in range(STEPS):
            env.action = cycle[t % len(cycle)]
            env.step()
            fit.append(float(np.mean(env.fitness)))
        per_episode.append(np.mean(fit))
    return np.mean(per_episode), np.std(per_episode, ddof=1) / np.sqrt(episodes)

rows = []
for a in range(1, K + 1):
    rows.append(((a,), *run((a,))))
    for b in range(1, K + 1):
        if a != b:
            rows.append(((a, b), *run((a, b))))
df = pd.DataFrame([("-".join(map(str, c)), len(c), m, se) for c, m, se in rows],
                  columns=["cycle", "n_drugs", "mean_fitness", "std_error"]).sort_values("mean_fitness")
os.makedirs("results/raw", exist_ok=True)
df.to_csv("results/raw/fixed_cycles.csv", index=False)
print(df.head(5).to_string(index=False))
print("random-drug reference is evaluated by run_experiment.py (policy 'naive').")
