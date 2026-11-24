"""Run RL vs random vs exact optimum on the 15 beta-lactam landscapes (N=4).

usage: python run_baseline.py SEED EPISODES TRAIN_FREQ DELAY MODE
  MODE = repo     repo's own delay option as shipped (Conv1D agent, acts on true state)
         state    delayed observation: stale genotype only (Dense agent)
         history  delayed observation: stale genotype + the drugs given since (Dense agent)
         decoy    same input size as history, drug slots filled with random drugs (control)
With DELAY=0, MODE=state is the no-delay reference for the Dense agent.
"""
import os, sys, time, random
import numpy as np
import pandas as pd
import tensorflow as tf
from evodm.exp import evol_deepmind

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
train_freq = int(sys.argv[3]) if len(sys.argv) > 3 else 1
delay = int(sys.argv[4]) if len(sys.argv) > 4 else 0
mode = sys.argv[5] if len(sys.argv) > 5 else "repo"
random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

kw = dict(episodes=episodes, N=4, num_drugs=15, mira=True, train_freq=train_freq)
if mode == "repo":
    kw.update(train_input="state_vector", delay=delay)
elif mode in ("state", "history", "decoy"):
    kw.update(train_input="delayed_" + mode, obs_delay=delay, delay=0)
else:
    sys.exit("MODE must be repo, state, history or decoy")

t = time.time()
out = evol_deepmind(**kw)
agents = {"RL": out[2], "naive": out[3], "DP": out[4]}

def per_episode(agent):
    d = {}
    for row in agent.master_memory:
        d.setdefault(row[0], []).append(float(np.mean(row[-1])))
    return np.array([np.mean(d[k]) for k in sorted(d)])

res = {k: per_episode(a) for k, a in agents.items()}
n = min(len(v) for v in res.values())
os.makedirs("results/raw", exist_ok=True)
df = pd.DataFrame({k: v[:n] for k, v in res.items()})
df.index.name = "episode"
df.to_csv(f"results/raw/{mode}_seed{seed}_ep{episodes}_tf{train_freq}_d{delay}.csv")

for k, v in res.items():
    print(k, "mean fitness, last 50 episodes:", round(float(np.mean(v[-50:])), 4), "episodes:", len(v))
print("seconds:", round(time.time() - t, 1))
