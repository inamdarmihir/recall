"""CLI entry point for recall cache management."""
from __future__ import annotations
import sys


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="recall — subagent result cache")
    sub = parser.add_subparsers(dest="cmd")

    lookup_p = sub.add_parser("lookup", help="Check cache for a key")
    lookup_p.add_argument("--subagent-type", required=True)
    lookup_p.add_argument("--system-prompt", default="")
    lookup_p.add_argument("--model", default="gpt-4o")
    lookup_p.add_argument("--tools-version", default="1.0.0")
    lookup_p.add_argument("--input", required=True)
    lookup_p.add_argument("--qdrant-url", default="http://localhost:6333")

    stats_p = sub.add_parser("stats", help="Print cache collection info")
    stats_p.add_argument("--qdrant-url", default="http://localhost:6333")

    args = parser.parse_args()

    if args.cmd == "lookup":
        from qdrant_client import QdrantClient
        from recall.store import RecallStore
        from recall.keys import compute_key
        client = QdrantClient(url=args.qdrant_url)
        store = RecallStore(client)
        key = compute_key(args.subagent_type, args.system_prompt, args.model, args.tools_version, args.input)
        result = store.lookup_exact(key)
        if result:
            print(f"HIT: {result}")
        else:
            print("MISS")

    elif args.cmd == "stats":
        from qdrant_client import QdrantClient
        client = QdrantClient(url=args.qdrant_url)
        info = client.get_collection("recall_cache")
        print(f"Collection: recall_cache — {info.points_count} entries")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
