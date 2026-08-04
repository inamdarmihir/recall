"""Shared typing helpers for Recall cache profiles and lookup results.

These types document the shape operators pass into
:class:`~recall.middleware.SubagentResultCacheMiddleware` and the payload
returned from :class:`~recall.store.SubagentCacheStore`. They are not
enforced at runtime — dictionaries still work — but they make the public
API easier to read in editors and for end-user setup.
"""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import UUID


class CacheProfile(TypedDict, total=False):
    """Per-subagent cache profile registered with the middleware.

    Required keys for a cache-eligible (``pure=True``) profile:

    - ``system_prompt``: full system prompt string for the subagent
    - ``model``: model id including provider pin
      (e.g. ``"anthropic:claude-haiku-4-5"``)
    - ``tools_version``: opaque version hash of the tool schemas exposed
      to the subagent — bump this when tools change so old entries miss
    - ``pure``: when ``True``, ``task`` calls for this type are cacheable

    Profiles with ``pure=False`` (or missing ``pure``) always bypass the
    cache. Keep the prompt/model/tools_version fields in sync with the
    matching entry in ``create_deep_agent(..., subagents=[...])``.
    """

    system_prompt: str
    model: str
    tools_version: str
    pure: bool


class CacheHit(TypedDict):
    """Exact-lookup hit returned by :meth:`SubagentCacheStore.lookup_exact`."""

    point_id: str | int | UUID
    payload: dict[str, Any]
    result: str
