"""Unified verification for all 5 optimization tasks.

Tests:
  1. GPU migration (device=auto → cuda)
  2. Batch processing speed (20 pairs < 2s)
  3. Cache collision safety (different inputs → different results)
  4. Thread safety (10 concurrent threads)
  5. Cache hit speedup (second call < 10% of first)

Usage:
    uv run python tests/verify_optimizations.py
"""
from __future__ import annotations
import sys
import io
import time
import threading
import os

# Fix Windows GBK encoding for emoji output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nli_scorer import NLIScorer


def main():
    print("=" * 60)
    print("  Unified Optimization Verification")
    print("=" * 60)

    scorer = NLIScorer(device='auto')
    print(f"NLI device: {scorer.device}")
    print()

    # ── Verify 1: GPU migration ──
    import torch
    if torch.cuda.is_available():
        assert scorer.device == 'cuda', f'Expected cuda, got {scorer.device}'
        print('[PASS] Verify 1: NLI on GPU')
    else:
        assert scorer.device == 'cpu', f'Expected cpu, got {scorer.device}'
        print('[PASS] Verify 1: NLI on CPU (no CUDA available)')

    # ── Verify 2: Batch processing speed ──
    pairs = [('Paris is in France.', 'France')] * 20
    t = time.time()
    results = scorer.score_batch(pairs)
    elapsed = time.time() - t
    assert len(results) == 20, f'Expected 20 results, got {len(results)}'
    assert elapsed < 2.0, f'Batch too slow: {elapsed:.2f}s'
    print(f'[PASS] Verify 2: 20 pairs batch = {elapsed:.3f}s')

    # ── Verify 3: Cache collision safety ──
    p1 = 'Berlin Wall fell in 1989' + 'x' * 200
    p2 = 'Berlin Wall fell in 1991' + 'x' * 200
    r1 = scorer.score_pair(p1, '1989')
    r2 = scorer.score_pair(p2, '1989')
    assert r1.entailment_score != r2.entailment_score, \
        f'Cache collision! p1={r1.entailment_score:.4f}, p2={r2.entailment_score:.4f}'
    print(f'[PASS] Verify 3: No cache collision '
          f'(p1_ent={r1.entailment_score:.3f}, p2_ent={r2.entailment_score:.3f})')

    # ── Verify 4: Thread safety ──
    errors = []
    def worker(i):
        try:
            scorer.score_pair(f'Unique sentence number {i} here.', 'answer')
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f'Thread safety errors: {errors}'
    print(f'[PASS] Verify 4: 10 concurrent threads, no errors')

    # ── Verify 5: Cache hit speedup ──
    test_premise = 'The capital of France is Paris.'
    test_hypothesis = 'Paris'

    # Clear cache for this pair to measure fresh
    key = scorer._make_cache_key(test_premise, test_hypothesis)
    with scorer._cache_lock:
        scorer._nli_cache.pop(key, None)

    t1 = time.time()
    scorer.score_pair(test_premise, test_hypothesis)
    first = time.time() - t1

    t2 = time.time()
    scorer.score_pair(test_premise, test_hypothesis)
    cached = time.time() - t2

    assert cached < first * 0.1 or first < 0.001, \
        f'Cache not faster: first={first*1000:.1f}ms, cached={cached*1000:.1f}ms'
    print(f'[PASS] Verify 5: Cache hit {cached*1000:.1f}ms vs '
          f'first {first*1000:.1f}ms')

    print()
    print(scorer.cache_stats())
    print()
    print('=' * 60)
    print('  All verifications passed')
    print('=' * 60)


if __name__ == '__main__':
    main()
