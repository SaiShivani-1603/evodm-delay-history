# Design freeze (written before any main run)

DRAFT: Sai to read, edit and commit this file before starting `scripts/run_all.sh`. After the first main run,
only typos may change; anything else is recorded as a deviation at the end of this file.

## Question
When the genotype reading is delayed, does telling the agent which drugs were given since that reading
restore performance, and is the benefit specific to the real drug record?

## Conditions (all use the same untuned hyperparameters, 500 episodes, `train_freq=1`)
- `state`: sees only the stale genotype distribution (control for the contribution).
- `history`: stale genotype plus the drugs given since (contribution).
- `decoy`: stale genotype plus random drugs, same input size as `history` (control for input size).
- `state` at delay 0: the no-delay reference with the same Dense network.
- `repo` at delay 0: the repository's own Conv1D agent at the same budget (matched-budget reproduction).
- Delays 1, 2 and 3. Reference policies: random drugs and the exact optimum, evaluated on the same protocol.

## Seeds
Seeds 10 to 14 for every condition; seeds 15 to 17 are added for the delay-2 conditions only.
Seeds 0 to 4 were used in pilots and are not reused.

## Primary outcome
Mean fitness over 50 greedy evaluation episodes at the training delay (no exploration, no learning), per
run; lower is better. Secondary: share of the optimum's benefit retained, (naive minus RL) / (naive minus DP).

## Hypotheses and decision rules (95% t-intervals on paired differences, pairing by seed)
- H1: at delays 2 and 3, history minus state is at most -0.15 and the interval excludes zero.
  (Delay 2 uses seeds 10 to 17, t = 2.365 with 7 degrees of freedom; delay 3 uses seeds 10 to 14, t = 2.776 with 4.)
- H2: at delays 2 and 3, history minus decoy is at most -0.10 and the interval excludes zero.
- H3: for a history agent trained at delay 2 and evaluated at staleness 1 and 3, history minus state (same
  training delay, same test staleness) is at most -0.10 with the interval excluding zero.
- Delay 1 is exploratory: reported with intervals, with no threshold.
A hypothesis that misses its threshold is reported as not supported. No threshold is changed after the runs.

## Secondary analyses (stated in advance)
- Learning curves for every condition (training fitness per episode block).
- What each agent learns: drug-use frequencies, action entropy, and agreement with the exact optimal policy,
  computed from the logged actions.
- Dense versus Conv1D at delay 0, reported as a limitation.
- Best fixed two-drug cycle (no readings), evaluated by scripts/fixed_cycles.py, reported beside every agent.
- Drug use, action entropy and agreement with the exact optimal policy, from the evaluation traces.

## Alternative explanations to test
Input size (decoy), too little training (learning curves), fewer stored transitions in the repository's own
delay option, seed variation (intervals and per-seed points).

## Compute
About 64 runs. One 500-episode run plus evaluation takes about 6 to 7 minutes in a batch of four on the
M1 MacBook Air (8 GB), so about 7 hours in total. To be measured and recorded.

## Deviations
(none yet)
