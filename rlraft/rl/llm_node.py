from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import urllib.error
import urllib.request
import random
from typing import Any

from rlraft.env import load_env
from rlraft.sim.sim import TIMEOUT_ARMS, NodeObservation


LAST_NODE_LLM_ERROR: str | None = None
LAST_NODE_LLM_KEY_SOURCE: str | None = None


@dataclass(frozen=True, slots=True)
class LLMNodeDecision:
    source: str
    action: str
    timeout_ms: float
    notes: str = ""


class LLMNodePolicyClient:
    """Direct LLM timeout policy for Raft nodes.

    This is deliberately not a reward-shaping advisor. Each node gives the LLM
    its local observation and receives a timeout arm. Decisions are cached by
    observation bucket so 50-node experiments do not spend one API call per
    election timer reset.
    """

    def __init__(self, require_llm: bool = False):
        self.require_llm = require_llm
        self.cache: dict[str, str] = {}

    def decide_from_sim_observation(
        self,
        observation: NodeObservation,
        rng: random.Random,
    ) -> LLMNodeDecision:
        payload = {
            "node_id": observation.node_id,
            "mean_rtt_ms": observation.mean_rtt_ms,
            "loss_rate": observation.loss_rate,
            "log_gap": observation.log_gap,
            "recent_split": observation.recent_split,
        }
        return self.decide(payload, rng)

    def decide(self, observation: dict[str, Any], rng: random.Random) -> LLMNodeDecision:
        key = _observation_key(observation)
        action = self.cache.get(key)
        source = "llm_cache" if action else "llm"
        if action is None:
            action = _request_llm_action(observation)
            if action is not None:
                self.cache[key] = action
        if action is None:
            if self.require_llm:
                raise RuntimeError(f"LLM node policy required but unavailable: {LAST_NODE_LLM_ERROR}")
            action = _fallback_action(observation)
            source = "deterministic_fallback"

        low, high = TIMEOUT_ARMS[action]
        return LLMNodeDecision(
            source=source,
            action=action,
            timeout_ms=rng.uniform(float(low), float(high)),
            notes="direct_node_timeout_decision",
        )


class LLMNodeTimeoutPolicy:
    def __init__(self, require_llm: bool = False):
        self.client = LLMNodePolicyClient(require_llm=require_llm)

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        return self.client.decide_from_sim_observation(observation, rng).timeout_ms


def check_llm_node_policy() -> dict[str, Any]:
    rng = random.Random(1)
    client = LLMNodePolicyClient(require_llm=False)
    decision = client.decide(
        {
            "node_id": 0,
            "mean_rtt_ms": 42.0,
            "loss_rate": 0.01,
            "log_gap": 0,
            "recent_split": False,
        },
        rng,
    )
    return {
        "source": decision.source,
        "action": decision.action,
        "timeout_ms": decision.timeout_ms,
        "openai_key_source": LAST_NODE_LLM_KEY_SOURCE,
        "llm_error": LAST_NODE_LLM_ERROR,
    }


def _request_llm_action(observation: dict[str, Any]) -> str | None:
    global LAST_NODE_LLM_ERROR, LAST_NODE_LLM_KEY_SOURCE
    LAST_NODE_LLM_ERROR = None
    LAST_NODE_LLM_KEY_SOURCE = None
    load_env()
    api_keys = [
        ("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")),
        ("OPENAI_API_KEY_2", os.environ.get("OPENAI_API_KEY_2")),
    ]
    api_keys = [(name, value) for name, value in api_keys if value]
    if not api_keys:
        LAST_NODE_LLM_ERROR = "missing_api_key"
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    body = {
        "model": model,
        "store": False,
        "temperature": 0.0,
        "instructions": (
            "You are one Raft node choosing only your own election timeout. "
            "Fast, reliable, fresh-log nodes should choose shorter timeouts. "
            "Slow, lossy, stale-log, or split-prone nodes should back off. "
            "Return only JSON with an action field."
        ),
        "input": json.dumps(
            {
                "local_observation": observation,
                "valid_actions": list(TIMEOUT_ARMS),
                "schema": {"action": "one of valid_actions", "notes": "short string"},
            },
            indent=2,
        ),
    }
    errors: list[str] = []
    for key_name, api_key in api_keys:
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = _extract_response_text(payload)
            parsed = json.loads(_extract_json_object(text))
            action = str(parsed.get("action", "")).strip()
            if action in TIMEOUT_ARMS:
                LAST_NODE_LLM_KEY_SOURCE = key_name
                return action
            errors.append(f"{key_name}:invalid_action:{action}")
        except urllib.error.HTTPError as exc:
            errors.append(f"{key_name}:http_{exc.code}:{_safe_error_body(exc)}")
            if exc.code not in {401, 403, 429}:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            errors.append(f"{key_name}:network_error:{exc}")
            break
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append(f"{key_name}:invalid_response:{exc}")
            break

    LAST_NODE_LLM_ERROR = " | ".join(errors) if errors else "unknown_api_failure"
    return None


def _fallback_action(observation: dict[str, Any]) -> str:
    rtt = min(float(observation.get("mean_rtt_ms", observation.get("estimated_rtt_ms", 200.0))) / 350.0, 1.5)
    loss = min(float(observation.get("loss_rate", 0.0)) / 0.22, 1.5)
    log_gap = min(float(observation.get("log_gap", 0.0)) / 6.0, 1.0)
    score = 0.62 * rtt + 0.28 * loss + 0.10 * log_gap
    if observation.get("recent_split") or observation.get("recent_split_vote"):
        score += 0.15
    if score < 0.16:
        return "very_short"
    if score < 0.30:
        return "short"
    if score < 0.52:
        return "medium"
    if score < 0.85:
        return "long"
    return "very_long"


def _observation_key(observation: dict[str, Any]) -> str:
    rtt = round(float(observation.get("mean_rtt_ms", observation.get("estimated_rtt_ms", 0.0))) / 25.0)
    loss = round(float(observation.get("loss_rate", 0.0)) / 0.025)
    log_gap = int(observation.get("log_gap", 0))
    split = bool(observation.get("recent_split", observation.get("recent_split_vote", False)))
    return f"rtt={rtt};loss={loss};log={log_gap};split={int(split)}"


def _extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    pieces: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                pieces.append(content["text"])
    return "\n".join(pieces)


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return "unreadable_error_body"
    body = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", body)
    body = re.sub(r"Bearer\\s+[A-Za-z0-9._-]+", "Bearer REDACTED", body)
    return body[:500]
