# RL-Raft Research Notes

## Grounding

Raft safety is not changed. The project only changes how nodes choose election
timeouts. Vote granting still follows Raft constraints: one vote per term,
majority winner, and candidate log freshness.

The implemented research question is:

> Can decentralized intelligent timeout policies reduce split votes and bias
> leadership toward better-connected nodes under heterogeneous network
> conditions?

## Current Method

- Simulator: deterministic discrete-event Raft election simulator.
- Process demo: local multi-process Raft cluster with dashboard controls.
- Baselines: static randomized Raft and a transparent Dynatune-like heuristic.
- Deep MARL: MAPPO centralized training/decentralized execution with rollout
  storage, GAE advantages, and PPO clipping.
- LLM-MAPPO: each node gets an LLM/fallback timeout prior inside the MAPPO
  rollout, so LLM behavior is part of the learned policy path rather than a
  separate advisor.
- LLM agents: pure prompt-based node agents remain as an ablation/demo mode.
- Legacy baseline: tabular Q-learning remains available as `--algorithm qlearning`.
- Safety: learning/LLM decisions affect election timing only, never vote rules.

## Important Correction

The previous tabular learned policy used a hand-coded quality offset during
execution. That made results partly heuristic. The offset has been removed from:

- `rlraft/sim/sim.py`
- `rlraft/rl/policy.py`

Any old result table based on that offset should be treated as obsolete. New
paper-quality results must be regenerated.

## LLM Role

The LLM is no longer framed as a reward-shaping advisor. The direct LLM mode is:

1. Each node observes only local information such as RTT estimate, log gap, and
   recent split/election signals.
2. In `llm_mappo`, the node asks the LLM for one timeout prior:
   `very_short`, `short`, `medium`, `long`, or `very_long`.
3. The prior biases the MAPPO actor logits during rollout collection and PPO
   updates.
4. The result is cached by observation bucket.
5. If API access fails, the node uses deterministic fallback and records the
   fallback source in the policy artifact.

This matches the intended claim better: nodes are intelligent agents, not a
central trainer receiving occasional LLM hints.

## MAPPO Role

The MAPPO implementation uses centralized training and decentralized execution:

- Actor input: local node observation plus optional LLM-prior logit bias.
- Critic input: local observation plus compact global cluster summary.
- Action: discrete election-timeout arm.
- Reward: failover success, fast election, low split votes, fewer competing
  candidates, and best-connected-node leadership.
- Rollout: repeated election rounds until success or retry exhaustion.
- Advantage estimator: GAE over per-node round trajectories.

This is a practical MAPPO-style implementation, not a formal proof of optimal
Dec-POMDP behavior.

## Required Ablations

A serious research run should compare:

- `static`
- `dynatune_like`
- `qlearning`
- `llm_mappo`
- `mappo` without LLM prior
- `llm_nodes`
- `llm_nodes` with forced fallback

Run each across 10-30 seeds at 50, 100, and 200 nodes. Report confidence
intervals for success rate, split votes, best-node win rate, failover time, and
candidate count.

## Honest Limitations

- This is a research simulator plus demo, not production Raft.
- Full log-replication correctness, persistence hardening, membership changes,
  snapshots, and crash recovery are still limited.
- The process dashboard is for demonstration. The deterministic simulator is
  still the source of controlled experimental claims.
- LLM calls are network/API dependent; fallback behavior must be reported
  separately from real LLM behavior.
- The latest local API check reached OpenAI, but both `.env` keys returned
  `insufficient_quota`; current smoke artifacts therefore record deterministic
  fallback priors, not successful paid LLM decisions.
