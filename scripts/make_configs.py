"""Write the configuration file for every experiment in the paper to configs/main/.

Design (frozen before the main runs): Dense network, 500 episodes, train_freq=1, untuned
hyperparameters identical for every condition. Seeds 10-14 for all conditions; seeds 15-17 are
added for the delay-2 conditions only (decided in advance, not after seeing results).
The repository's own agent (mode repo, Conv1D, no delay) is run at the same budget as the
matched-budget reproduction.
"""
import json, os

OUT = "configs/main"
EPISODES, TRAIN_FREQ, EVAL_EPISODES, EVAL_STALE = 500, 1, 50, [0, 1, 2, 3, 4]
os.makedirs(OUT, exist_ok=True)

conditions = [("state", 0)]
for d in (1, 2, 3):
    conditions += [("state", d), ("history", d), ("decoy", d)]

def write(mode, d, s):
    rid = f"main_{mode}_d{d}_s{s}"
    cfg = dict(id=rid, mode=mode, delay=d, seed=s, episodes=EPISODES, train_freq=TRAIN_FREQ,
               eval_episodes=EVAL_EPISODES, eval_stale=EVAL_STALE)
    json.dump(cfg, open(f"{OUT}/{rid}.json", "w"), indent=2)

for mode, d in conditions:
    seeds = range(10, 18) if d == 2 else range(10, 15)
    for s in seeds:
        write(mode, d, s)
for s in range(10, 15):
    write("repo", 0, s)
print(len(os.listdir(OUT)), "config files in", OUT)
