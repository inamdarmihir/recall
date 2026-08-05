# Configuration

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `RECALL_TOOLS_VERSION` | `1.0.0` | Version string; change to invalidate cache |
| `RECALL_MAX_AGE_SECONDS` | `86400` | TTL for cached results (default: 1 day) |

## SubagentResultCacheMiddleware Parameters

```python
SubagentResultCacheMiddleware(
    store: RecallStore,
    profiles: dict[str, dict],   # {"subagent_type": {"purity": SubagentPurity}}
    tools_version: str = "1.0.0",
)
```

## RecallStore Parameters

```python
RecallStore(
    client: QdrantClient,
    collection: str = "recall_cache",
)
```

## SubagentPurity Enum

```python
class SubagentPurity(str, Enum):
    PURE = "pure"            # deterministic, safe to cache
    IMPURE = "impure"        # non-deterministic or side-effectful
    CONDITIONAL = "conditional"  # opt-in by caller
```

## Cache Key Components

The cache key is SHA-256 of the canonical JSON of:

```json
{
  "subagent_type": "researcher",
  "system_prompt": "You are a researcher.",
  "model": "gpt-4o",
  "tools_version": "1.0.0",
  "task_input": {"task": "summarise X"}
}
```

All keys are sorted before hashing, so input dict ordering does not affect the key.
