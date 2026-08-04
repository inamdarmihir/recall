"""Cache-key design: deterministic hashing of subagent profile + task inputs.

The key is the correctness surface of Recall. Under-specified keys produce
false hits; over-specified keys produce unnecessary misses.

A cache key captures everything that can change a subagent's output:

.. code-block:: text

    k = Hash(θ_type, θ_prompt, θ_model, θ_tools_version, x_canonical)

where ``x_canonical`` includes the task description **and** content hashes
of any named source files. Path-only identity is not enough — editing a
file must change the key so the old entry is never looked up.

See ``docs/ARTICLE.md`` §6 for the full design rationale.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

# File-like extensions recognized when scanning backtick tokens in descriptions.
_FILE_SUFFIXES = (".ts", ".py", ".md", ".tsx", ".js", ".json", ".toml", ".yaml", ".yml")


def canonicalize(obj: object) -> str:
    """Return a deterministic serialization suitable for cache-key material.

    Rules:

    - ``dict`` — keys sorted, values recursively canonicalized, compact JSON
    - ``list`` / ``tuple`` — element-wise canonicalize, then compact JSON
    - ``str`` — NFC-normalized Unicode with collapsed whitespace
    - numbers / bool / ``None`` — JSON literals
    - anything else — stringified, then canonicalized as a string

    Two semantically identical payloads that differ only in key order,
    trailing whitespace, or Unicode normalization hash identically.
    """
    if isinstance(obj, dict):
        items = {k: canonicalize(v) for k, v in sorted(obj.items())}
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return json.dumps(
            [canonicalize(v) for v in obj],
            separators=(",", ":"),
            ensure_ascii=False,
        )
    if isinstance(obj, str):
        return " ".join(unicodedata.normalize("NFC", obj).split())
    if isinstance(obj, (int, float, bool)) or obj is None:
        return json.dumps(obj)
    return canonicalize(str(obj))


def normalize_path(path: str) -> str:
    """Normalize filesystem paths to POSIX form for stable hashing.

    Collapses ``.`` segments (``docs/./a.md`` → ``docs/a.md``) without
    resolving symlinks or forcing absolute paths — callers decide the
    VFS root separately via ``vfs_root`` on the middleware.
    """
    return str(PurePosixPath(path))


def profile_prefix(subagent_type: str, profile: dict[str, Any]) -> str:
    """Hash the static profile tuple ``(type, prompt, model, tools_version)``.

    Memoize this at middleware init if profiles are static for the session —
    recomputing SHA-256 over a multi-kilobyte system prompt on every
    dispatch is wasteful.

    Parameters
    ----------
    subagent_type:
        Subagent name / profile identifier (matches ``task.subagent_type``).
    profile:
        Dict with ``system_prompt``, ``model``, and ``tools_version`` keys.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest (no ``sha256:`` prefix — that is added
        by :func:`compute_key` on the full material).
    """
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
    """Collect file references from a ``task`` tool-call argument dict.

    Preference order:

    1. Explicit ``files`` or ``paths`` lists on the tool args (extension
       beyond the stock Deep Agents ``TaskToolSchema``, useful when the
       orchestrator or REPL passes structured file lists).
    2. Backtick-quoted tokens in ``description`` that look like file paths
       (contain ``/`` or end with a known source suffix).

    Paths discovered only inside the subagent's own tool loop are invisible
    here — that is the provenance open problem discussed in the article §13.
    Returns an ordered, de-duplicated list of POSIX-normalized paths.
    """
    refs: list[str] = []
    for path in args.get("files") or args.get("paths") or []:
        refs.append(normalize_path(str(path)))
    for match in re.findall(r"`([^`]+)`", args.get("description", "")):
        if "/" in match or match.endswith(_FILE_SUFFIXES):
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
    """Compute the content-addressed cache key for one ``task`` invocation.

    File content hashes are folded into ``x_canonical`` so a path-only
    identity cannot false-hit after an edit. The returned string is
    ``"sha256:"`` plus the hex digest.

    Parameters
    ----------
    subagent_type:
        Profile / subagent name.
    profile:
        Cache profile dict (see :class:`~recall.types.CacheProfile`).
    args:
        ``task`` tool args — at minimum ``description``; optionally
        ``files`` / ``paths``.
    content_hash_fn:
        Callable ``path -> "sha256:..."`` used to fingerprint each source
        ref. The middleware supplies a VFS-rooted hasher.
    """
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
