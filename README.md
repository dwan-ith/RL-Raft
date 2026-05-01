# RL-Raft

RL-Raft is a Raft leader-election research prototype. Nodes independently learn
election timeout behavior from repeated practice elections. The goal is simple:
fast, well-connected nodes should learn to start elections earlier, while slow,
lossy, or stale-log nodes should learn to wait.

Raft safety rules are not learned or weakened. The learned policy only controls
when a node starts an election; majority voting, one vote per term, and log
freshness remain Raft rules.

## LLM + MARL Training

Training uses an LLM advisor by default. API keys are loaded from `.env`
automatically:

```text
OPENAI_API_KEY=...
OPENAI_API_KEY_2=...
OPENAI_MODEL=gpt-4o-mini
```

The advisor tries `OPENAI_API_KEY`, then `OPENAI_API_KEY_2`. If both are
unavailable or over quota, training falls back to deterministic guidance and
records that in `advisor_history`.

Check LLM access without printing secrets:

```powershell
python -m rlraft.cli check-llm
```

Train with LLM-default behavior:

```powershell
python -m rlraft.cli train --episodes 35000 --nodes 50 --output runs/policies/learned_policy.json
```

Force a real LLM run and fail if fallback would be used:

```powershell
python -m rlraft.cli train --episodes 35000 --nodes 50 --output runs/policies/learned_policy.json --require-llm
```

## Experiments

```powershell
python -m rlraft.cli sim-compare --nodes 50 --episodes 1000 --seed 92
```

This compares:

- `static`: vanilla randomized Raft election timeouts.
- `dynatune_like`: measurement-based adaptive timeout heuristic.
- `learned`: tabular MARL timeout policy with LLM reward-shaping advisor by default.

## Live Demo

```powershell
python -m rlraft.cli start --nodes 50 --policy learned --port 8000
```

Open `http://127.0.0.1:8000`.

## Project Structure

- `rlraft/core/`: process-based Raft demo engine, network hub, vote rules.
- `rlraft/sim/`: deterministic event-driven Raft election simulator and experiments.
- `rlraft/rl/`: MARL trainer, LLM advisor, learned timeout policy loader.
- `rlraft/web/`: dashboard server.
- `frontend/`: dashboard assets.
- `tests/`: Raft rules, simulator, trainer, `.env`, and LLM fallback tests.

## Current Caveat

The `.env` keys are detected, but the latest local API check returned
`insufficient_quota` for both keys. Until quota/billing is fixed, runs use
deterministic fallback guidance unless `--require-llm` is used, in which case
training fails loudly.
