from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from rlraft.env import load_env

LAST_LLM_ERROR: str | None = None
LAST_LLM_KEY_SOURCE: str | None = None


@dataclass(slots=True)
class AdvisorGuidance:
    source: str = "deterministic"
    split_penalty: float = 3.5
    best_leader_bonus: float = 2.2
    non_best_leader_penalty: float = 0.8
    candidate_penalty: float = 0.04
    failover_scale_ms: float = 700.0
    early_quality_threshold: float = 0.14
    elite_quality_threshold: float = 0.12
    split_early_penalty: float = 0.9
    split_late_bonus: float = 0.5
    notes: str = "Deterministic fallback guidance."

    def clipped(self) -> "AdvisorGuidance":
        return AdvisorGuidance(
            source=self.source,
            split_penalty=_clip(self.split_penalty, 1.0, 8.0),
            best_leader_bonus=_clip(self.best_leader_bonus, 0.5, 6.0),
            non_best_leader_penalty=_clip(self.non_best_leader_penalty, 0.0, 4.0),
            candidate_penalty=_clip(self.candidate_penalty, 0.0, 0.25),
            failover_scale_ms=_clip(self.failover_scale_ms, 250.0, 1800.0),
            early_quality_threshold=_clip(self.early_quality_threshold, 0.04, 0.35),
            elite_quality_threshold=_clip(self.elite_quality_threshold, 0.02, 0.25),
            split_early_penalty=_clip(self.split_early_penalty, 0.0, 3.0),
            split_late_bonus=_clip(self.split_late_bonus, 0.0, 2.0),
            notes=self.notes[:600],
        )


def deterministic_guidance(metrics: dict[str, float] | None = None) -> AdvisorGuidance:
    guidance = AdvisorGuidance()
    metrics = metrics or {}
    split_rate = metrics.get("split_vote_rate", 0.0)
    best_rate = metrics.get("best_node_win_rate", 0.0)
    if split_rate > 0.20:
        guidance.split_penalty = 5.2
        guidance.split_early_penalty = 1.3
        guidance.split_late_bonus = 0.8
        guidance.candidate_penalty = 0.07
        guidance.notes = "Fallback increased split/candidate penalties."
    if best_rate < 0.55:
        guidance.best_leader_bonus = 3.8
        guidance.non_best_leader_penalty = 1.4
        guidance.early_quality_threshold = 0.12
        guidance.elite_quality_threshold = 0.10
        guidance.notes = "Fallback increased best-leader pressure."
    return guidance


def get_default_guidance(
    metrics: dict[str, float] | None = None,
    greedy_policy: dict[str, str] | None = None,
    advisor: str = "llm",
) -> AdvisorGuidance:
    if advisor == "off":
        return AdvisorGuidance(source="off", notes="Advisor disabled.")
    if advisor == "deterministic":
        return deterministic_guidance(metrics)
    llm = request_llm_guidance(metrics or {}, greedy_policy or {})
    if llm is not None:
        return llm
    fallback = deterministic_guidance(metrics)
    fallback.source = "deterministic_fallback"
    fallback.notes = "LLM unavailable; " + fallback.notes
    return fallback


def request_llm_guidance(
    metrics: dict[str, float],
    greedy_policy: dict[str, str],
) -> AdvisorGuidance | None:
    global LAST_LLM_ERROR, LAST_LLM_KEY_SOURCE
    LAST_LLM_ERROR = None
    LAST_LLM_KEY_SOURCE = None
    load_env()
    api_keys = [
        ("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")),
        ("OPENAI_API_KEY_2", os.environ.get("OPENAI_API_KEY_2")),
    ]
    api_keys = [(name, value) for name, value in api_keys if value]
    if not api_keys:
        LAST_LLM_ERROR = "missing_api_key"
        return None

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    prompt = _build_prompt(metrics, greedy_policy)
    body = {
        "model": model,
        "store": False,
        "temperature": 0.2,
        "instructions": (
            "You are an expert in Raft leader election and multi-agent "
            "reinforcement learning. Return only compact JSON."
        ),
        "input": prompt,
    }
    payload = None
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
            LAST_LLM_KEY_SOURCE = key_name
            break
        except urllib.error.HTTPError as exc:
            errors.append(f"{key_name}:http_{exc.code}:{_safe_error_body(exc)}")
            if exc.code not in {401, 403, 429}:
                break
        except urllib.error.URLError as exc:
            errors.append(f"{key_name}:url_error:{exc.reason}")
            break
        except TimeoutError:
            errors.append(f"{key_name}:timeout")
            break
        except json.JSONDecodeError:
            errors.append(f"{key_name}:invalid_json_response")
            break
    if payload is None:
        LAST_LLM_ERROR = " | ".join(errors) if errors else "unknown_api_failure"
        return None

    text = _extract_response_text(payload)
    if not text:
        LAST_LLM_ERROR = "empty_response_text"
        return None
    try:
        parsed = json.loads(_extract_json_object(text))
    except json.JSONDecodeError:
        LAST_LLM_ERROR = "invalid_guidance_json"
        return None
    return _guidance_from_json(parsed)


def _build_prompt(metrics: dict[str, float], greedy_policy: dict[str, str]) -> str:
    schema = {
        "split_penalty": "float 1.0-8.0",
        "best_leader_bonus": "float 0.5-6.0",
        "non_best_leader_penalty": "float 0.0-4.0",
        "candidate_penalty": "float 0.0-0.25",
        "failover_scale_ms": "float 250-1800",
        "early_quality_threshold": "float 0.04-0.35",
        "elite_quality_threshold": "float 0.02-0.25",
        "split_early_penalty": "float 0.0-3.0",
        "split_late_bonus": "float 0.0-2.0",
        "notes": "short string",
    }
    return json.dumps(
        {
            "task": (
                "Advise reward shaping for RL-Raft. Nodes learn election timeouts. "
                "Goal: reduce split votes, keep failover fast, and make best-connected "
                "nodes win more often. Do not change Raft safety."
            ),
            "current_metrics": metrics,
            "current_greedy_policy_sample": dict(list(greedy_policy.items())[:30]),
            "allowed_json_schema": schema,
        },
        indent=2,
    )


def _extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    pieces: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    pieces.append(text)
    return "\n".join(pieces)


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _guidance_from_json(data: dict[str, Any]) -> AdvisorGuidance:
    defaults = AdvisorGuidance()
    guidance = AdvisorGuidance(
        source="llm",
        split_penalty=float(data.get("split_penalty", defaults.split_penalty)),
        best_leader_bonus=float(data.get("best_leader_bonus", defaults.best_leader_bonus)),
        non_best_leader_penalty=float(
            data.get("non_best_leader_penalty", defaults.non_best_leader_penalty)
        ),
        candidate_penalty=float(data.get("candidate_penalty", defaults.candidate_penalty)),
        failover_scale_ms=float(data.get("failover_scale_ms", defaults.failover_scale_ms)),
        early_quality_threshold=float(
            data.get("early_quality_threshold", defaults.early_quality_threshold)
        ),
        elite_quality_threshold=float(
            data.get("elite_quality_threshold", defaults.elite_quality_threshold)
        ),
        split_early_penalty=float(
            data.get("split_early_penalty", defaults.split_early_penalty)
        ),
        split_late_bonus=float(data.get("split_late_bonus", defaults.split_late_bonus)),
        notes=str(data.get("notes", "")),
    )
    return guidance.clipped()


def guidance_to_dict(guidance: AdvisorGuidance) -> dict[str, Any]:
    return asdict(guidance.clipped())


def _clip(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return "unreadable_error_body"
    body = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", body)
    body = re.sub(r"Bearer\\s+[A-Za-z0-9._-]+", "Bearer REDACTED", body)
    return body[:500]
