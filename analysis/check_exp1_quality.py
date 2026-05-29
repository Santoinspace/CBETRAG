"""Exp1 quality check: identify damaged samples (empty answers / empty cache).

Damage = vLLM concurrent timeouts → empty responses → cached as "".
NOT damage = low CS scores (Edge Support scores naturally vary).

Criteria for damaged qids:
  1. method == 'CBET' AND answer is empty/whitespace
  (These are the samples where the LLM returned "" due to timeout)

Usage:
    # Step 1: scan only (no changes)
    uv run python analysis/check_exp1_quality.py

    # Step 2: fix (remove damaged results + purge empty cache)
    uv run python analysis/check_exp1_quality.py --fix

    # Step 3: re-run damaged samples only (resume logic skips completed)
    uv run python experiments/run_exp1_main.py \
        --datasets hotpotqa musique --n_samples 500 --workers 2
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def extract_damaged_qids(results_path: str) -> list[str]:
    """Find qids where any method returned an empty answer.

    These are the samples where vLLM returned empty responses due to
    concurrent request timeouts, and the empty response got cached.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    damaged_qids: set[str] = set()
    for r in data:
        answer = r.get("answer", "").strip()
        if not answer:
            damaged_qids.add(r["qid"])

    return sorted(damaged_qids)


def remove_damaged_results(results_path: str, damaged_qids: list[str]) -> int:
    """Remove all entries for damaged qids from the results JSON.

    Removes CBET + baselines for each damaged qid so the resume logic
    in run_exp1_main.py treats them as fully pending and re-runs all methods.
    Returns the number of deleted entries.
    """
    qid_set = set(damaged_qids)
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    original_len = len(data)
    kept = [r for r in data if r["qid"] not in qid_set]
    removed = original_len - len(kept)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    return removed


def purge_empty_cache_entries(cache_dir: str = ".llm_cache") -> int:
    """Scan .llm_cache/, delete entries with empty response text.

    These are invalid cache entries written when vLLM returned empty
    responses due to concurrent timeouts. Removing them forces a fresh
    API call on the next run.
    """
    purged = 0
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"  Cache directory {cache_dir} not found")
        return 0

    for cache_file in cache_path.glob("*.json"):
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            text = data.get("text", "MISSING")
            if text == "" or not text.strip():
                cache_file.unlink()
                purged += 1
        except Exception:
            # Corrupted JSON — also purge
            cache_file.unlink()
            purged += 1

    return purged


def main():
    parser = argparse.ArgumentParser(description="Exp1 quality check & repair")
    parser.add_argument("--results", default="experiments/results/exp1_main.json")
    parser.add_argument("--cache_dir", default=".llm_cache")
    parser.add_argument("--fix", action="store_true",
                        help="Actually remove damaged results and purge cache")
    args = parser.parse_args()

    print("=" * 70)
    print("Exp1 Quality Check — Damaged Sample Detection & Repair")
    print("=" * 70)

    # ── Step 1: identify damaged qids (empty answers) ──────────────────────
    print(f"\n[1/3] Scanning {args.results} ...")
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)
    cbet = [r for r in data if r["method"] == "CBET"]
    print(f"  Total entries: {total_entries}, CBET: {len(cbet)}")

    damaged_qids = extract_damaged_qids(args.results)
    print(f"  Damaged qids (any method empty answer): {len(damaged_qids)}")

    # Show details per damaged qid
    for qid in damaged_qids:
        entries = [r for r in data if r["qid"] == qid]
        methods_empty = [r["method"] for r in entries if not r.get("answer", "").strip()]
        methods_ok = [r["method"] for r in entries if r.get("answer", "").strip()]
        ds = entries[0].get("dataset", "?") if entries else "?"
        print(f"    qid={qid}  dataset={ds}  "
              f"empty=[{','.join(methods_empty)}]  ok=[{','.join(methods_ok)}]")

    # ── Step 2: count empty cache entries ──────────────────────────────────
    print(f"\n[2/3] Scanning {args.cache_dir} for empty response entries ...")
    cache_path = Path(args.cache_dir)
    total_cache = len(list(cache_path.glob("*.json"))) if cache_path.exists() else 0
    empty_cache = 0
    for cf in cache_path.glob("*.json"):
        try:
            d = json.loads(cf.read_text(encoding="utf-8"))
            t = d.get("text", "MISSING")
            if t == "" or not t.strip():
                empty_cache += 1
        except Exception:
            empty_cache += 1
    print(f"  Total cache: {total_cache}, empty responses: {empty_cache}")

    if not damaged_qids and empty_cache == 0:
        print("\n  No damage detected. Nothing to do.")
        return

    # ── Step 3: fix or report ──────────────────────────────────────────────
    if not args.fix:
        print(f"\n[3/3] DRY RUN — no changes made.")
        print(f"  To apply fix, run:")
        print(f"    uv run python analysis/check_exp1_quality.py --fix")
        print(f"\n  This will:")
        if damaged_qids:
            print(f"    1. Remove {len(damaged_qids)} damaged qids from results "
                  f"(all methods)")
        if empty_cache:
            print(f"    2. Purge {empty_cache} empty cache entries")
        print(f"\n  Then re-run:")
        print(f"    uv run python experiments/run_exp1_main.py \\")
        print(f"        --datasets hotpotqa musique --n_samples 500 --workers 2")
        return

    print(f"\n[3/3] Applying fix ...")

    # Remove damaged results
    if damaged_qids:
        removed = remove_damaged_results(args.results, damaged_qids)
        with open(args.results, encoding="utf-8") as f:
            remaining = len(json.load(f))
        print(f"  Removed {removed} entries ({len(damaged_qids)} damaged qids)")
        print(f"  Remaining: {remaining} entries")

    # Purge empty cache
    purged = purge_empty_cache_entries(args.cache_dir)
    print(f"  Purged {purged} empty cache entries")

    # Summary
    ds_counts = {}
    for r in data:
        if r["qid"] in set(damaged_qids):
            ds = r["dataset"]
            ds_counts[ds] = ds_counts.get(ds, 0) + 1
    print(f"\n  Deleted {len(damaged_qids)} damaged qids:")
    for ds, n in sorted(ds_counts.items()):
        print(f"    {ds}: {n} entries")
    print(f"  Remaining {total_entries - len(damaged_qids) * 4} results preserved")

    print(f"\n{'=' * 70}")
    print(f"Done. Re-run with resume (auto-skips completed samples):")
    print(f"  uv run python experiments/run_exp1_main.py \\")
    print(f"      --datasets hotpotqa musique --n_samples 500 --workers 2")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
