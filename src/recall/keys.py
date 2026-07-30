"""Cache-key design: deterministic hashing of subagent profile + task inputs."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any


def canonicalize(obj: object) -> str:
    """Deterministic serialization for cache-key material."""
    if isinstance(obj, dict):
        items = {k: canonicalize(v) for k, v in sorted(obj.items())}
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return json.dumps([canonicalize(v) for v in obj], separators=(",", ":"), ensure_ascii=False)
    if isinstance(obj, str):
        return " ".join(unicodedata.normalize("NFC", obj).split())
    if isinstance(obj, (int, float, bool)) or obj is None:
        return json.dumps(obj)
    return canonicalize(str(obj))


def normalize_path(path: str) -> str:
    """Normalize filesystem paths to POSIX form for stable hashing."""
    return str(PurePosixPath(path))


def profile_prefix(subagent_type: str, profile: dict[str, Any]) -> str:
    """SHA-256 over the static (θ_type, θ_prompt, θ_model, θ_tools_version) tuple."""
    material = canonicalize(
        {
            "type": subagent_type,
            "prompt": profile["system_prompt"],
            "model": profile["model"],
            "tools": profile["tools_version"],
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def extract_source_refs(args: dict[str, Any]) -> list[str]:
    """Prefer explicit `files`/`paths`; else scan backtick-quoted file-like tokens."""
    refs: list[str] = []
    for p in args.get("files") or args.get("paths") or []:
        refs.append(normalize_path(str(p)))
    for match in re.findall(r"`([^`]+)`", args.get("description", "")):
        if "/" in match or match.endswith((".ts", ".py", ".md", ".tsx", ".js")):
            refs.append(normalize_path(match))
    seen: set[str] = set()
    ordered: list[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered


def compute_key(
    subagent_type: str,
    profile: dict[str, Any],
    args: dict[str, Any],
    content_hash_fn: Callable[[str], str],
) -> str:
    """Fold file content hashes into x_canonical so path-only identity cannot false-hit."""
    source_refs = extract_source_refs(args)
    content_hashes = {ref: content_hash_fn(ref) for ref in source_refs}
    x_canonical = canonicalize(
        {
            "description": args.get("description", ""),
            "files": {normalize_path(p): h for p, h in sorted(content_hashes.items())},
        }
    )
    material = f"{profile_prefix(subagent_type, profile)}:{x_canonical}"
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
