"""Content-addressed cache key computation for subagent dispatches."""
from __future__ import annotations
import hashlib
import json
from enum import Enum
from typing import Any


class SubagentPurity(str, Enum):
    """Whether a subagent's output is deterministic given its inputs."""
    PURE = "pure"           # safe to cache; same input → same output
    IMPURE = "impure"       # non-deterministic or has side effects; never cache
    CONDITIONAL = "conditional"  # may be cached but caller must opt in explicitly


def canonicalize(value: Any) -> str:
    """Produce a stable, canonical JSON string for any JSON-serializable value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_key(
    subagent_type: str,
    system_prompt: str,
    model: str,
    tools_version: str,
    task_input: Any,
) -> str:
    """
    Compute a SHA-256 content-addressed cache key.
    Key space: (subagent_type, system_prompt, model, tools_version, canonical_task_input)
    """
    payload = canonicalize({
        "subagent_type": subagent_type,
        "system_prompt": system_prompt,
        "model": model,
        "tools_version": tools_version,
        "task_input": task_input,
    })
    return hashlib.sha256(payload.encode()).hexdigest()


def classify_purity(subagent_type: str, profiles: dict[str, dict]) -> SubagentPurity:
    """Look up purity classification from a profiles registry."""
    profile = profiles.get(subagent_type, {})
    raw = profile.get("purity", SubagentPurity.IMPURE)
    if isinstance(raw, SubagentPurity):
        return raw
    return SubagentPurity(raw)
