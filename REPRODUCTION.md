# Reproduction note (Stage 2)

Source paper: Weaver, King, Maltas, Scott. *Reinforcement learning informs optimal treatment strategies to limit antibiotic resistance.* PNAS 121(16): e2303165121 (2024), doi 10.1073/pnas.2303165121 (preprint: bioRxiv 10.1101/2023.01.12.523765).
Source code: https://github.com/DavisWeaver/evo_dm, commit `d975114` (GPL-3.0).
Everything below is computed from the files in `results/raw/` and `logs/` of this repository.

## What is reproduced

Claims in the published paper (PNAS 121(16), Results sections 2.1 and 2.5; Figs. 2A, 2B and 6B), checked against the PNAS PDF:
1. The agent that sees the genotype (RL-genotype) beats the random drug policy in all 100 replicates.
2. The distribution of its performance "nearly overlapped" the exact optimum from backward induction (the paper gives no numeric fraction).
3. Both RL agents "learned effective drug cycling policies after about 100 episodes" and improved minimally to episode 500.
4. Measurement delay: performance is equivalent to the base case for d <= 1 and falls to "worse than or similar to the random condition" for d > 1 (RL-genotype only; the exact optimum and random policies are not delayed).

Paper setup (Methods and Results): 500 episodes of 20 steps, 100 replicates per condition, reward 1 minus fitness, RL-genotype network of two Conv1D layers, max pooling, a 28-neuron dense layer and a linear output, mean squared error loss with a slowly updated target network. The paper does not state the learning rate, batch size, discount, exploration schedule or training frequency in the main text. Figure code is in a second repository, DavisWeaver/rl_cycling.

Setting reproduced: 15 beta-lactam landscapes (`mira=True`), N = 4 (16 genotypes), 20-step episodes, repo's default RL-genotype network (Conv1D), `train_freq=1`.

## Scaled reproduction: what was reduced, and why

| Item | Paper | Mine | Reason |
|---|---|---|---|
| Training episodes | 500 | 200 | Time on a laptop CPU; one 200-episode run is about 300 s alone |
| Replicates | 100 | 5 (seeds 0 to 4) | Same |
| Hardware | not stated in main text | MacBook Air M1, 8 GB, CPU only | Available hardware |

## Results (mean fitness over the last 50 episodes; lower is better)

| Policy | Mean over 5 seeds | 95% interval (t, 4 d.f.) |
|---|---|---|
| RL agent | 0.253 | +/- 0.049 |
| Random (naive) | 0.625 | +/- 0.009 |
| Exact optimum (DP) | 0.052 | +/- 0.007 |
| Share of the optimum's benefit retained, (naive - RL)/(naive - DP) | 65% | +/- 9 percentage points |

- RL beat random on 5 of 5 seeds. Per-seed RL: 0.224, 0.301, 0.212, 0.288, 0.241.
- Learning curve (mean RL fitness per 50-episode block, seed 0): 0.641, 0.460, 0.239, 0.223; every seed is still falling in the last block.
- Delay (repo's own `delay=3`, seeds 0 to 3): RL fitness 0.528 against 0.256 at delay 0 on the same seeds; two of four seeds end near random level (0.657 and 0.635 against about 0.62).

## Agreement with the paper

- RL beats random on every replicate: agrees (5 of 5).
- The agent is far better than random but does not match the optimum: consistent in direction, but weaker than "nearly overlaps".
- Measurement delay hurts: agrees in direction. My delay-3 result (0.528, n = 4) is milder than "worse than or similar to random", with two of four seeds at random level.
- Note on the paper's own labelling: the text cites Fig. 6C for delay, while the Fig. 6 caption describes delay in panel B. I cite the caption (6B). The text says delays d in {0, 5} were tested; the panel shows delays 0 to 5.
- Note on what "delay" means: the paper describes the learner as "using genotype information from time t to inform an action taken on time t+5", i.e. acting on stale information. My reading of the repository code is that its `delay` option shifts the training tuples while the agent still acts on the true current state. I have not confirmed this against the authors' rl_cycling scripts, so the delay comparison above is stated for the repository option as shipped.

## Differences and my explanations

1. **Fewer episodes (200 against 500).** All five seeds are still improving at episode 200, so the 65% figure probably understates the converged agent. Untested until the planned 500-episode run of the same agent.
2. **Fewer replicates (5 against 100).** Wide intervals; claims limited to this scale.
3. **Training frequency.** The repo's `evol_deepmind` defaults to `train_freq=100`, which gives about 100 weight updates in total. At 500 episodes, seed 0 this agent scores 0.724 against random 0.626, i.e. worse than random. Using `train_freq=1` fixes it (0.224). The paper's main text does not state the training frequency, so I do not know which setting the authors used.
4. **Metric.** I score the last 50 training episodes, which are near-greedy because epsilon decays to its minimum. The paper's evaluation protocol is not fully described in the text I could read.
5. **Software and platform.** I run Python 3.9.25 and TensorFlow 2.11 on Apple silicon. The repository's profiling file was produced on Windows with Python 3.9, so platform differences are possible.
6. **Seeds.** Python, NumPy and TensorFlow seeds are fixed per run. Repeated runs can still differ slightly, which is expected.

## Compute record

- One 200-episode run alone: about 300 s (seed 0).
- Four runs in parallel: 512 to 516 s each, 8 min 42 s in total, about 2.2 min per run effective.
- Hardware: MacBook Air, Apple M1, 8 GB memory, CPU only (no GPU), macOS 27.2.1 (build 26B131).
- Software: Python 3.9.25 in conda env `antibio`; TensorFlow (tensorflow-macos) 2.11.0, Keras 2.11.0, NumPy 1.23.5, pandas 2.3.3, SciPy 1.13.1, pytest 8.4.2, evodm 1.1.2 (installed from this repository). Full lists: `environment.yml` (conda) and `requirements-frozen.txt` (pip, the authoritative list for the pip-installed packages), plus `ENVIRONMENT.txt` (date, OS, hardware, Python, git commit).
- Repository state when the environment was recorded: commit `9b1ffeac91fc0c0e7a227f8460cba55f29900b68` on branch `delay-history`, captured 27 Nov 2026 21:29 GMT. The 200-episode baseline runs were made at earlier commits of the same branch, before the observation modes were added to `run_baseline.py`. The matched-budget 500-episode rerun of the repo's own agent (`MODE=repo`) at the submission commit will confirm the baseline is unchanged.
