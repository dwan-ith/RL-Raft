# RL-Raft Research Notes

## Grounding

Raft safety is not changed. The project only changes how nodes choose election
timeouts. Vote granting still follows the Raft constraints: one vote per term,
majority winner, and candidate log freshness.

The research question implemented here is:

> Can decentralized learned timeout behavior reduce split votes and bias
> leadership toward better-connected nodes under heterogeneous network
> conditions?

## What Was Removed

The earlier learned-policy artifact was not good enough because it effectively
encoded node identity. That is not a real RL-Raft claim. The upgraded trainer no
longer hardcodes a winning node. Each training/evaluation episode samples a new
heterogeneous topology, computes the best-connected node from RTT/loss/log
freshness, and evaluates whether the learned local policy makes that node win.

## Current Method

- Simulator: deterministic discrete-event Raft election simulator.
- Baselines: static randomized Raft and a Dynatune-like measurement heuristic.
- Learned policy: independent Q-learning with parameter sharing.
- LLM role: the default trainer calls an LLM advisor for reward/timeout-shaping
  guidance. If no API key/network/model is available, deterministic guidance is
  used as a fallback and recorded in the policy artifact.
- API keys are loaded from `.env` automatically. The advisor tries
  `OPENAI_API_KEY` first, then `OPENAI_API_KEY_2`.
- The latest local LLM diagnostic reached OpenAI but returned
  `insufficient_quota` for both configured keys, so local runs currently use
  deterministic fallback unless quota/billing is fixed.
- Local observations: mean RTT, loss estimate, log gap, recent split signal.
- Output: timeout value in milliseconds.
- Safety: unchanged Raft voting rules; learning affects liveness/performance only.

## Latest 50-Node Results

Command:

```powershell
python -m rlraft.cli sim-compare --nodes 50 --episodes 500 --output-dir runs --seed 92
```

Results:

| Policy | Success | Split votes/failover | Best-node win rate | Avg failover |
|---|---:|---:|---:|---:|
| static | 0.826 | 2.236 | 0.102 | 3350.5 ms |
| dynatune_like | 1.000 | 0.110 | 0.478 | 481.3 ms |
| learned | 1.000 | 0.090 | 0.848 | 537.7 ms |

100-node stress comparison:

| Policy | Success | Split votes/failover | Best-node win rate | Avg failover |
|---|---:|---:|---:|---:|
| static | 0.572 | 3.680 | 0.074 | 5325.7 ms |
| dynatune_like | 1.000 | 0.172 | 0.348 | 570.4 ms |
| learned | 1.000 | 0.122 | 0.824 | 552.9 ms |

Interpretation:

- Static Raft is fast when lucky, but split votes are frequent at 50 nodes.
- The Dynatune-like baseline is strong, but less leader-aware.
- Learned RL-Raft sharply reduces split retries versus static Raft and chooses
  the best-connected node far more often than both baselines.
- At 200 nodes the learned 50-node policy still succeeds, but split retries rise;
  that scale should be treated as stress-test territory unless retrained/tuned.

## Honest Limitations

- This is a research simulator plus demo, not production Raft.
- The learned policy is not MAPPO yet; it is a transparent tabular MARL learner
  with LLM-assisted reward shaping by default.
- The process dashboard is for visualization. The deterministic simulator is the
  source of experimental claims.
- Full log-replication correctness, persistence, membership changes, and
  production-grade recovery are out of scope for the current prototype.
