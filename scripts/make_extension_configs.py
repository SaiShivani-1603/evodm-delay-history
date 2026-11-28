"""Write the configuration files for the extension experiments (decided after the main results).

These are EXPLORATORY and are reported as such: they were added after the main batch had been
analysed, to test three objections to its results. The hypotheses in DESIGN_FREEZE.md and the main
analysis are unchanged.

  configs/ext_long/   E1  longer training (1500 episodes): does history close the gap to the best
                          fixed cycle if given more training?   history/state at delay 2 and the
                          repository agent at delay 0, seeds 10-14
  configs/ext_conv/   E2  convolutional front end (the repository agent's network) for the new
                          agents: is the result an artefact of the Dense network?   state, history,
                          decoy at delay 2, 500 episodes, seeds 10-14
  configs/ext_seeds/  E3  seeds 15-17 for the delay 1 and delay 3 conditions, so every condition has
                          8 seeds, 500 episodes, Dense, otherwise identical to the main batch
"""
import json, os

EVAL_EPISODES, EVAL_STALE = 50, [0, 1, 2, 3, 4]

def write(folder, mode, d, s, episodes, network="dense", tag=""):
    os.makedirs(folder, exist_ok=True)
    rid = f"{os.path.basename(folder)}_{mode}_d{d}_s{s}"
    cfg = dict(id=rid, mode=mode, delay=d, seed=s, episodes=episodes, train_freq=1,
               eval_episodes=EVAL_EPISODES, eval_stale=EVAL_STALE, network=network)
    json.dump(cfg, open(f"{folder}/{rid}.json", "w"), indent=2)

for s in range(10, 15):
    for mode, d in (("history", 2), ("state", 2), ("repo", 0)):
        write("configs/ext_long", mode, d, s, 1500)
    for mode in ("state", "history", "decoy"):
        write("configs/ext_conv", mode, 2, s, 500, network="conv")
for s in range(15, 18):
    for d in (1, 3):
        for mode in ("state", "history", "decoy"):
            write("configs/ext_seeds", mode, d, s, 500)
for f in ("ext_long", "ext_conv", "ext_seeds"):
    print(f, len(os.listdir(f"configs/{f}")), "configs")
