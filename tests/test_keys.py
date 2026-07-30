"""Unit tests for cache-key canonicalization and content hashing."""

from __future__ import annotations

from recall.keys import (
    canonicalize,
    compute_key,
    extract_source_refs,
    normalize_path,
    profile_prefix,
)

PROFILE = {
    "system_prompt": "Summarize the document.",
    "model": "anthropic:claude-haiku-4-5",
    "tools_version": "a3f9c2d1",
    "pure": True,
}


def test_canonicalize_sorts_dict_keys() -> None:
    a = canonicalize({"b": 1, "a": 2})
    b = canonicalize({"a": 2, "b": 1})
    assert a == b


def test_canonicalize_normalizes_whitespace() -> None:
    assert canonicalize("hello   world\n") == canonicalize("hello world")


def test_normalize_path_posix() -> None:
    assert normalize_path("docs/./a.md") == "docs/a.md"
    assert normalize_path("/abs/path/file.py") == "/abs/path/file.py"


def test_extract_source_refs_from_files_and_backticks() -> None:
    args = {
        "description": "Summarize structural changes in `docs/intro.md`",
        "files": ["docs/outro.md"],
    }
    refs = extract_source_refs(args)
    assert refs == ["docs/outro.md", "docs/intro.md"]


def test_compute_key_stable_and_content_sensitive(tmp_path) -> None:
    path = tmp_path / "page.md"
    path.write_text("# Hello\n", encoding="utf-8")

    def hasher(ref: str) -> str:
        target = tmp_path / ref
        return f"sha256:{target.read_bytes().hex()}" if target.exists() else "sha256:missing"

    args = {"description": f"Summarize `{path.name}`", "files": [path.name]}
    key1 = compute_key("doc-summarizer", PROFILE, args, hasher)
    key2 = compute_key("doc-summarizer", PROFILE, args, hasher)
    assert key1 == key2
    assert key1.startswith("sha256:")

    path.write_text("# Hello world\n", encoding="utf-8")
    key3 = compute_key("doc-summarizer", PROFILE, args, hasher)
    assert key3 != key1


def test_profile_prefix_changes_with_model() -> None:
    other = {**PROFILE, "model": "anthropic:claude-sonnet-5"}
    assert profile_prefix("doc-summarizer", PROFILE) != profile_prefix("doc-summarizer", other)
