"""Parallel experiment runner — ThreadPoolExecutor for I/O-bound vLLM calls.

vLLM runs in the cloud (network I/O). While one sample waits for the cloud
response, the local GPU can process NLI for another sample concurrently.

Usage:
    from src.experiment_runner import run_experiment_parallel
    results = run_experiment_parallel(
        questions=questions,
        method_fn=lambda q: run_single(q, llm, nli, retriever),
        max_workers=4,
        checkpoint_fn=lambda r: save_json(r, path),
    )
"""
from __future__ import annotations
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_experiment_parallel(
    questions: list,
    method_fn: callable,
    max_workers: int = 4,
    checkpoint_every: int = 10,
    checkpoint_fn: callable | None = None,
    desc: str = "samples",
) -> list[dict]:
    """Process questions in parallel with periodic checkpointing.

    Args:
        questions: list of Question objects (or dicts) to process
        method_fn: callable(question) -> dict result
        max_workers: 2-3 conservative, 4-6 recommended, >8 may hit rate limits
        checkpoint_every: save checkpoint every N completed samples
        checkpoint_fn: callable(results_list) for checkpointing
        desc: label for progress printing

    Returns:
        List of result dicts, sorted by qid for deterministic output.
    """
    results: list[dict] = []
    results_lock = threading.Lock()
    counter = [0]
    counter_lock = threading.Lock()
    t_start = time.time()

    def process_one(q):
        try:
            return method_fn(q)
        except Exception as e:
            qid = getattr(q, 'qid', None) or (q.get('qid', '?') if isinstance(q, dict) else '?')
            print(f"  [WARN] sample {qid} failed: {e}")
            return {"qid": str(qid), "error": str(e), "em": 0, "f1": 0.0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, q): q for q in questions}

        for future in as_completed(futures):
            result = future.result()

            with results_lock:
                results.append(result)

            with counter_lock:
                counter[0] += 1
                n = counter[0]

            if n % 10 == 0 or n == len(questions):
                elapsed = time.time() - t_start
                rate = n / elapsed if elapsed > 0 else 0
                print(f"  [{n}/{len(questions)}] {desc} done "
                      f"({rate:.1f}/s, elapsed {elapsed:.0f}s)")

            if checkpoint_fn and n % checkpoint_every == 0:
                with results_lock:
                    sorted_snapshot = sorted(results, key=lambda r: str(r.get("qid", "")))
                    checkpoint_fn(sorted_snapshot)

    # Sort by qid for deterministic output (as_completed returns unordered)
    results.sort(key=lambda r: str(r.get("qid", "")))

    elapsed = time.time() - t_start
    print(f"  Completed {len(results)} {desc} in {elapsed:.0f}s "
          f"({len(results)/elapsed:.1f}/s with {max_workers} workers)")

    return results
