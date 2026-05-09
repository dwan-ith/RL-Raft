from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from rlraft.rl.llm_node import LLMNodePolicyClient
from rlraft.sim.sim import (
    FEATURE_NAMES,
    TIMEOUT_ARMS,
    NodeCondition,
    NodeObservation,
    TimeoutPolicy,
    generate_conditions,
    observation_quality_score,
    observations_from_conditions,
    simulate_election_round,
    simulate_failover,
)

GLOBAL_FEATURE_NAMES = [
    "node_count_norm",
    "alive_fraction",
    "mean_rtt_norm",
    "mean_loss",
    "mean_log_gap_norm",
    "recent_split",
    "round_index_norm",
]


@dataclass(slots=True)
class MAPPOTrainingResult:
    episodes: int
    nodes: int
    success_rate: float
    split_vote_rate: float
    best_node_win_rate: float
    average_failover_ms: float
    policy_path: str


@dataclass(slots=True)
class MAPPOConfig:
    learning_rate: float = 3e-4
    batch_episodes: int = 64
    ppo_epochs: int = 4
    clip_ratio: float = 0.20
    entropy_coef: float = 0.015
    value_coef: float = 0.50
    gamma: float = 0.96
    gae_lambda: float = 0.90
    max_rounds: int = 6
    llm_prior_strength: float = 1.40
    llm_bc_coef: float = 0.04


def train_mappo_policy(
    episodes: int = 8000,
    nodes: int = 50,
    output_path: str = "runs/policies/llm_mappo_policy.json",
    seed: int = 7,
    learning_rate: float = 3e-4,
    batch_episodes: int = 64,
    ppo_epochs: int = 4,
    clip_ratio: float = 0.20,
    entropy_coef: float = 0.015,
    value_coef: float = 0.50,
    gamma: float = 0.96,
    gae_lambda: float = 0.90,
    max_rounds: int = 6,
    use_llm_prior: bool = True,
    require_llm: bool = False,
    llm_prior_strength: float = 1.40,
    llm_bc_coef: float = 0.04,
    eval_episodes: int = 1000,
) -> MAPPOTrainingResult:
    """Train MAPPO over repeated Raft election-round trajectories.

    This is centralized training with decentralized execution. Each node actor
    sees only its local observation. The critic sees local observation plus a
    compact global summary. If `use_llm_prior` is true, each node also receives
    an LLM-derived action prior that biases the actor distribution and is stored
    in the rollout for PPO updates and ablation.
    """

    torch, _nn, optim, _categorical = _torch_deps()
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    config = MAPPOConfig(
        learning_rate=learning_rate,
        batch_episodes=batch_episodes,
        ppo_epochs=ppo_epochs,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        gamma=gamma,
        gae_lambda=gae_lambda,
        max_rounds=max_rounds,
        llm_prior_strength=llm_prior_strength,
        llm_bc_coef=llm_bc_coef,
    )

    model = ActorCritic(
        local_dim=len(FEATURE_NAMES),
        global_dim=len(GLOBAL_FEATURE_NAMES),
        action_dim=len(TIMEOUT_ARMS),
        hidden_dim=128,
    )
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    llm_client = LLMNodePolicyClient(require_llm=require_llm) if use_llm_prior else None
    llm_counts: dict[str, int] = {}
    buffer: list[dict[str, Any]] = []

    for episode in range(episodes):
        conditions = generate_conditions(nodes, rng)
        episode_records, source_counts = _collect_episode_rollout(
            model=model,
            conditions=conditions,
            rng=rng,
            config=config,
            llm_client=llm_client,
        )
        for source, count in source_counts.items():
            llm_counts[source] = llm_counts.get(source, 0) + count
        buffer.extend(episode_records)

        if (episode + 1) % config.batch_episodes == 0:
            _ppo_update(model, optimizer, buffer, config)
            buffer = []

    if buffer:
        _ppo_update(model, optimizer, buffer, config)

    eval_policy = MAPPOTimeoutPolicy.from_model(
        model,
        use_llm_prior=use_llm_prior,
        llm_prior_strength=config.llm_prior_strength,
        require_llm=False,
    )
    eval_metrics = evaluate_mappo_policy(eval_policy, nodes=nodes, seed=seed + 100_000, episodes=eval_episodes)
    artifact = {
        "policy_type": "mappo_timeout_policy",
        "algorithm": "llm_mappo_ctde_gae" if use_llm_prior else "mappo_ctde_gae",
        "episodes": episodes,
        "nodes": nodes,
        "seed": seed,
        "features": FEATURE_NAMES,
        "global_features": GLOBAL_FEATURE_NAMES,
        "arms": TIMEOUT_ARMS,
        "actor_critic": _state_dict_to_json(model.state_dict()),
        "hidden_dim": 128,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "max_rounds": config.max_rounds,
        "use_llm_prior": use_llm_prior,
        "require_llm": require_llm,
        "llm_prior_strength": config.llm_prior_strength,
        "llm_bc_coef": config.llm_bc_coef,
        "llm_source_counts": llm_counts,
        "evaluation_metrics": eval_metrics,
        "research_note": (
            "MAPPO with rollout storage, PPO clipping, centralized critic, GAE "
            "advantages, and decentralized actor execution. When enabled, LLM "
            "node priors are part of the policy distribution instead of a separate "
            "advisor path."
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return MAPPOTrainingResult(
        episodes=episodes,
        nodes=nodes,
        success_rate=eval_metrics["success_rate"],
        split_vote_rate=eval_metrics["split_vote_rate"],
        best_node_win_rate=eval_metrics["best_node_win_rate"],
        average_failover_ms=eval_metrics["average_failover_ms"],
        policy_path=str(path),
    )


def evaluate_mappo_policy(
    policy_or_model: Any,
    nodes: int = 50,
    seed: int = 17,
    episodes: int = 1000,
) -> dict[str, float]:
    rng = random.Random(seed)
    policy = (
        policy_or_model
        if isinstance(policy_or_model, MAPPOTimeoutPolicy)
        else MAPPOTimeoutPolicy.from_model(policy_or_model)
    )
    successes = split_votes = best_wins = 0
    total_time = total_rounds = total_candidates = 0.0
    for _ in range(episodes):
        conditions = generate_conditions(nodes, rng)
        result = simulate_failover(conditions, policy, rng)
        successes += int(result.success)
        split_votes += result.split_votes
        best_wins += int(result.best_node_won)
        total_time += result.total_time_ms
        total_rounds += result.rounds
        total_candidates += result.candidates_started
    return {
        "success_rate": successes / episodes,
        "split_vote_rate": split_votes / episodes,
        "best_node_win_rate": best_wins / episodes,
        "average_failover_ms": total_time / episodes,
        "average_rounds": total_rounds / episodes,
        "average_candidates_started": total_candidates / episodes,
    }


class MAPPOTimeoutPolicy(TimeoutPolicy):
    def __init__(
        self,
        model: Any,
        greedy: bool = True,
        use_llm_prior: bool = False,
        llm_prior_strength: float = 1.40,
        require_llm: bool = False,
    ):
        self.model = model
        self.greedy = greedy
        self.use_llm_prior = use_llm_prior
        self.llm_prior_strength = llm_prior_strength
        self.llm_client = LLMNodePolicyClient(require_llm=require_llm) if use_llm_prior else None

    @classmethod
    def from_artifact(cls, path: str) -> "MAPPOTimeoutPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("policy_type") != "mappo_timeout_policy":
            raise ValueError(f"unsupported MAPPO artifact: {data.get('policy_type')}")
        model = ActorCritic(
            local_dim=len(data.get("features", FEATURE_NAMES)),
            global_dim=len(data.get("global_features", GLOBAL_FEATURE_NAMES)),
            action_dim=len(data.get("arms", TIMEOUT_ARMS)),
            hidden_dim=int(data.get("hidden_dim", 128)),
        )
        model.load_state_dict(_state_dict_from_json(data["actor_critic"]))
        model.eval()
        return cls(
            model,
            use_llm_prior=bool(data.get("use_llm_prior", False)),
            llm_prior_strength=float(data.get("llm_prior_strength", 1.40)),
            require_llm=False,
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        use_llm_prior: bool = False,
        llm_prior_strength: float = 1.40,
        require_llm: bool = False,
    ) -> "MAPPOTimeoutPolicy":
        model.eval()
        return cls(
            model,
            use_llm_prior=use_llm_prior,
            llm_prior_strength=llm_prior_strength,
            require_llm=require_llm,
        )

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        torch, _nn, _optim, Categorical = _torch_deps()
        local = torch.tensor(observation.features(), dtype=torch.float32).unsqueeze(0)
        global_zeros = torch.zeros((1, len(GLOBAL_FEATURE_NAMES)), dtype=torch.float32)
        with torch.no_grad():
            logits, _value = self.model(local, global_zeros)
            prior_action = None
            if self.llm_client is not None:
                decision = self.llm_client.decide_from_sim_observation(observation, rng)
                prior_action = _action_index(decision.action)
            logits = _apply_prior_to_logits(logits, prior_action, self.llm_prior_strength)
            if self.greedy:
                action_index = int(torch.argmax(logits, dim=-1).item())
            else:
                action_index = int(Categorical(logits=logits).sample().item())
        arm_name = list(TIMEOUT_ARMS)[action_index]
        low, high = TIMEOUT_ARMS[arm_name]
        return rng.uniform(float(low), float(high))


def ActorCritic(local_dim: int, global_dim: int, action_dim: int, hidden_dim: int = 128):
    torch, nn, _optim, _categorical = _torch_deps()

    class _ActorCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(local_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(local_dim + global_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, local_obs: Any, global_obs: Any) -> tuple[Any, Any]:
            logits = self.actor(local_obs)
            value = self.critic(torch.cat([local_obs, global_obs], dim=-1))
            return logits, value

    return _ActorCritic()


def _collect_episode_rollout(
    model: Any,
    conditions: list[NodeCondition],
    rng: random.Random,
    config: MAPPOConfig,
    llm_client: LLMNodePolicyClient | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    torch, _nn, _optim, Categorical = _torch_deps()
    records: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    recent_split = False
    best = _safe_best_node(conditions)

    for round_index in range(config.max_rounds):
        observations = observations_from_conditions(conditions, recent_split=recent_split)
        global_features = _global_features(conditions, recent_split, round_index, config.max_rounds)
        fixed_timeouts: dict[int, float] = {}
        round_records: list[dict[str, Any]] = []

        for node_id, observation in observations.items():
            if not conditions[node_id].alive:
                continue
            local_tensor = torch.tensor(observation.features(), dtype=torch.float32).unsqueeze(0)
            global_tensor = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
            logits, value = model(local_tensor, global_tensor)
            prior_action = None
            prior_source = "none"
            if llm_client is not None:
                decision = llm_client.decide_from_sim_observation(observation, rng)
                prior_action = _action_index(decision.action)
                prior_source = decision.source
                source_counts[prior_source] = source_counts.get(prior_source, 0) + 1
            biased_logits = _apply_prior_to_logits(logits, prior_action, config.llm_prior_strength)
            dist = Categorical(logits=biased_logits)
            action = dist.sample()
            action_index = int(action.item())
            arm_name = list(TIMEOUT_ARMS)[action_index]
            low, high = TIMEOUT_ARMS[arm_name]
            fixed_timeouts[node_id] = rng.uniform(low, high)
            round_records.append(
                {
                    "local": observation.features(),
                    "global": global_features,
                    "action": action_index,
                    "old_log_prob": float(dist.log_prob(action).item()),
                    "value": float(value.squeeze(-1).item()),
                    "node_id": node_id,
                    "observation": observation,
                    "prior_action": prior_action if prior_action is not None else -1,
                    "prior_source": prior_source,
                    "active_mask": 1.0,
                    "done": False,
                    "reward": 0.0,
                }
            )

        result = _simulate_with_fixed_timeouts(conditions, fixed_timeouts, rng, recent_split)
        done = result.success or round_index == config.max_rounds - 1
        global_reward = _mappo_global_reward(result, final_failure=done and not result.success)
        for record in round_records:
            record["reward"] = global_reward + _mappo_local_reward(
                node_id=record["node_id"],
                observation=record["observation"],
                action_index=record["action"],
                leader_id=result.leader_id,
                best_node_id=best,
            )
            record["done"] = done
        records.extend(round_records)
        if result.success:
            break
        recent_split = True

    _add_gae(records, config.gamma, config.gae_lambda)
    return records, source_counts


def _ppo_update(
    model: Any,
    optimizer: Any,
    records: list[dict[str, Any]],
    config: MAPPOConfig,
) -> None:
    if not records:
        return
    torch, _nn, _optim, Categorical = _torch_deps()
    locals_tensor = torch.tensor([r["local"] for r in records], dtype=torch.float32)
    globals_tensor = torch.tensor([r["global"] for r in records], dtype=torch.float32)
    actions = torch.tensor([r["action"] for r in records], dtype=torch.long)
    old_log_probs = torch.tensor([r["old_log_prob"] for r in records], dtype=torch.float32)
    returns = torch.tensor([r["return"] for r in records], dtype=torch.float32)
    advantages = torch.tensor([r["advantage"] for r in records], dtype=torch.float32)
    prior_actions = torch.tensor([r["prior_action"] for r in records], dtype=torch.long)
    active_mask = torch.tensor([r["active_mask"] for r in records], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    for _ in range(config.ppo_epochs):
        logits, values = model(locals_tensor, globals_tensor)
        logits = _apply_prior_batch(logits, prior_actions, config.llm_prior_strength)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages
        policy_loss = -(torch.min(unclipped, clipped) * active_mask).sum() / active_mask.sum().clamp_min(1.0)
        value_loss = ((values.squeeze(-1) - returns).pow(2) * active_mask).sum() / active_mask.sum().clamp_min(1.0)
        entropy_loss = -(dist.entropy() * active_mask).sum() / active_mask.sum().clamp_min(1.0)
        llm_bc_loss = _llm_behavior_clone_loss(logits, prior_actions, active_mask)
        loss = (
            policy_loss
            + config.value_coef * value_loss
            + config.entropy_coef * entropy_loss
            + config.llm_bc_coef * llm_bc_loss
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _add_gae(records: list[dict[str, Any]], gamma: float, gae_lambda: float) -> None:
    by_node: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_node.setdefault(record["node_id"], []).append(record)
    for node_records in by_node.values():
        gae = 0.0
        next_value = 0.0
        for record in reversed(node_records):
            mask = 0.0 if record["done"] else 1.0
            delta = record["reward"] + gamma * next_value * mask - record["value"]
            gae = delta + gamma * gae_lambda * mask * gae
            record["advantage"] = gae
            record["return"] = gae + record["value"]
            next_value = record["value"]


def _simulate_with_fixed_timeouts(
    conditions: list[NodeCondition],
    fixed_timeouts: dict[int, float],
    rng: random.Random,
    recent_split: bool,
):
    class FixedPolicy:
        def timeout_ms(self, observation: NodeObservation, _rng: random.Random) -> float:
            return fixed_timeouts[observation.node_id]

    return simulate_election_round(conditions, FixedPolicy(), rng, recent_split=recent_split)


def _global_features(
    conditions: list[NodeCondition],
    recent_split: bool,
    round_index: int = 0,
    max_rounds: int = 6,
) -> list[float]:
    alive = [condition for condition in conditions if condition.alive]
    count = max(len(conditions), 1)
    sample = alive or conditions
    return [
        min(count / 200.0, 1.5),
        len(alive) / count,
        sum(c.mean_rtt_ms for c in sample) / len(sample) / 350.0,
        sum(c.loss_rate for c in sample) / len(sample),
        sum(c.log_gap for c in sample) / len(sample) / 8.0,
        1.0 if recent_split else 0.0,
        round_index / max(max_rounds - 1, 1),
    ]


def _mappo_global_reward(result: Any, final_failure: bool = False) -> float:
    reward = 3.0 if result.success else -1.5
    if final_failure:
        reward -= 4.0
    reward -= result.election_time_ms / 900.0
    reward -= 3.2 if result.split_vote else 0.0
    reward -= max(0, result.candidates_started - 1) * 0.05
    if result.success:
        reward += 3.4 if result.leader_id == result.best_node_id else -1.0
    return reward


def _mappo_local_reward(
    node_id: int,
    observation: NodeObservation,
    action_index: int,
    leader_id: int | None,
    best_node_id: int,
) -> float:
    score = observation_quality_score(observation)
    early = action_index <= 1
    late = action_index >= 3
    reward = 0.0
    if node_id == leader_id:
        reward += 2.0 if node_id == best_node_id else -1.3
    elif score < 0.16 and early:
        reward += 0.7
    elif score > 0.34 and late:
        reward += 0.7
    elif score > 0.34 and early:
        reward -= 0.8
    if observation.recent_split and early:
        reward -= 0.8
    return reward


def _apply_prior_to_logits(logits: Any, prior_action: int | None, strength: float) -> Any:
    if prior_action is None or prior_action < 0:
        return logits
    adjusted = logits.clone()
    adjusted[..., prior_action] += strength
    return adjusted


def _apply_prior_batch(logits: Any, prior_actions: Any, strength: float) -> Any:
    adjusted = logits.clone()
    valid = prior_actions >= 0
    if bool(valid.any()):
        rows = valid.nonzero(as_tuple=True)[0]
        adjusted[rows, prior_actions[rows]] += strength
    return adjusted


def _llm_behavior_clone_loss(logits: Any, prior_actions: Any, active_mask: Any) -> Any:
    torch, _nn, _optim, _categorical = _torch_deps()
    valid = (prior_actions >= 0) & (active_mask > 0)
    if not bool(valid.any()):
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
    return torch.nn.functional.cross_entropy(logits[valid], prior_actions[valid])


def _action_index(action: str) -> int:
    return list(TIMEOUT_ARMS).index(action)


def _safe_best_node(conditions: list[NodeCondition]) -> int:
    alive = [condition for condition in conditions if condition.alive]
    return min(
        alive or conditions,
        key=lambda c: (c.mean_rtt_ms + 850.0 * c.loss_rate + 35.0 * c.log_gap, c.node_id),
    ).node_id


def _state_dict_to_json(state_dict: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().tolist() for name, tensor in state_dict.items()}


def _state_dict_from_json(data: dict[str, Any]) -> dict[str, Any]:
    torch, _nn, _optim, _categorical = _torch_deps()
    return {name: torch.tensor(value, dtype=torch.float32) for name, value in data.items()}


def _torch_deps():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.distributions import Categorical
    except ImportError as exc:
        raise RuntimeError(
            "MAPPO training requires PyTorch. Install torch before running "
            "`python -m rlraft.cli train --algorithm llm_mappo`."
        ) from exc
    return torch, nn, optim, Categorical
