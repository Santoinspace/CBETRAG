"""Task 8: Evaluation and result aggregation.

Reads CBET and baseline result JSONs, computes all metrics,
and prints a paper-ready comparison table.

Usage:
    python analysis/evaluate_all.py --results_dir experiments/results/
    python analysis/evaluate_all.py --dataset hotpotqa
"""
import argparse
import json
import os
from pathlib import Path


# ── metric helpers ────────────────────────────────────────────────────────────

def _em(pred: str, gold: str) -> float:
    return float(pred.strip().lower() == gold.strip().lower())


def _f1(pred: str, gold: str) -> float:
    p_tok = pred.strip().lower().split()
    g_tok = gold.strip().lower().split()
    if not p_tok or not g_tok:
        return 0.0
    common = set(p_tok) & set(g_tok)
    if not common:
        return 0.0
    prec = len(common) / len(p_tok)
    rec  = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


# ── per-log LM call estimator ─────────────────────────────────────────────────
# Each iteration: 1 DAG extraction + n_branches × (1 probe + 1 claim_extract + 1 answer)
# + 1 final answer = 1 + iterations × (dag_size × 3) + 1

def _estimate_lm_calls(log: dict) -> int:
    dag_size   = log.get("dag_size", 1)
    iterations = log.get("iterations", 1)
    return 1 + iterations * (dag_size * 3) + 1


# ── aggregate one result file ─────────────────────────────────────────────────

def aggregate(logs: list[dict]) -> dict:
    n = len(logs)
    if n == 0:
        return {}

    em_scores  = [log.get("em",  _em(log.get("answer",""), log.get("gold_answer",""))) for log in logs]
    f1_scores  = [log.get("f1",  _f1(log.get("answer",""), log.get("gold_answer",""))) for log in logs]
    iters      = [log.get("iterations", 1) for log in logs]
    cs_scores  = [log.get("final_cs", 0.0) for log in logs]
    lm_calls   = [log.get("total_lm_calls", _estimate_lm_calls(log)) for log in logs]
    ret_calls  = [log.get("retrieval_calls", log.get("iterations", 1)) for log in logs]
    tokens     = [log.get("total_tokens_consumed", 0) for log in logs]
    early_stop = [log.get("early_stopped", False) for log in logs]

    conflict_rate = sum(1 for l in logs if l.get("conflicts_detected")) / n
    override_rate = sum(1 for l in logs if l.get("overrides_triggered")) / n
    noisy_rate    = sum(1 for l in logs if l.get("noisy_evicted"))       / n
    early_stop_rate = sum(early_stop) / n

    # Contains% metric: answer contains gold answer (substring match)
    contains_count = 0
    for log in logs:
        pred = log.get("answer", "").strip().lower()
        gold = log.get("gold_answer", "").strip().lower()
        if gold and gold in pred:
            contains_count += 1
    contains_rate = contains_count / n if n > 0 else 0.0

    # DAG telemetry: success rate, avg branches, fallback rate, avg hop count
    dag_success_keys = ("dag_success",)
    dag_fb_keys = ("dag_fallback",)
    dag_br_keys = ("dag_branches", "dag_size")
    dag_hop_keys = ("dag_hop_count",)

    dag_success_count = 0
    dag_fallback_count = 0
    dag_branches_list = []
    dag_hop_list = []
    for log in logs:
        # dag_success: True if DAG extraction succeeded
        if log.get("dag_success") is True or (log.get("dag_success") is None and not log.get("dag_fallback", False)):
            dag_success_count += 1
        # dag_fallback: True if extraction fell back to single-node
        if log.get("dag_fallback", False):
            dag_fallback_count += 1
        # branch count
        for key in dag_br_keys:
            if key in log:
                dag_branches_list.append(log[key])
                break
        # hop count
        for key in dag_hop_keys:
            if key in log:
                dag_hop_list.append(log[key])
                break

    dag_success_rate = dag_success_count / n if n > 0 else 0.0
    dag_fallback_rate = dag_fallback_count / n if n > 0 else 0.0
    avg_dag_branches = sum(dag_branches_list) / len(dag_branches_list) if dag_branches_list else 0.0
    avg_dag_hop = sum(dag_hop_list) / len(dag_hop_list) if dag_hop_list else 0.0

    return {
        "n":                        n,
        "em":                       100 * sum(em_scores) / n,
        "f1":                       100 * sum(f1_scores) / n,
        "contains":                 100 * contains_rate,
        "avg_retrieval_rounds":     sum(iters)    / n,
        "avg_lm_calls":             sum(lm_calls) / n,
        "avg_retrieval_calls":      sum(ret_calls) / n,
        "avg_tokens_consumed":      sum(tokens)   / n,
        "avg_cs_at_stop":           sum(cs_scores) / n,
        "conflict_detected_rate":   100 * conflict_rate,
        "override_triggered_rate":  100 * override_rate,
        "noisy_branch_evicted_rate":100 * noisy_rate,
        "early_stop_rate":          100 * early_stop_rate,
        "dag_success_rate":         100 * dag_success_rate,
        "avg_dag_branches":         avg_dag_branches,
        "dag_fallback_rate":        100 * dag_fallback_rate,
        "avg_dag_hop_count":        avg_dag_hop,
    }


# ── load helpers ──────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _find_results(results_dir: str, dataset: str | None) -> dict[str, dict]:
    """Return {method_label: metrics} for all result files in results_dir."""
    methods: dict[str, dict] = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        name = p.stem  # e.g. cbet_hotpotqa, ablation_full, baseline_DRAGIN_hotpotqa
        if dataset and dataset not in name:
            continue
        try:
            logs = _load(str(p))
            metrics = aggregate(logs)
            if metrics:
                methods[name] = metrics
        except Exception as e:
            print(f"[WARN] Could not load {p}: {e}")
    return methods


# ── table printer ─────────────────────────────────────────────────────────────

_BASELINE_ORDER = ["DRAGIN", "SeaKR", "FLARE", "AdaptiveRAG", "Adaptive_Rag"]

# Known baseline EM/F1 from AdaRAGUE paper (placeholders until run_baselines.sh completes)
_ADARAGUE_PAPER = {
    "hotpotqa": {
        "DRAGIN":      {"em": 42.1, "f1": 51.3, "avg_retrieval_rounds": 2.1, "avg_lm_calls": 2.3},
        "SeaKR":       {"em": 43.8, "f1": 53.0, "avg_retrieval_rounds": 1.9, "avg_lm_calls": 2.1},
        "FLARE":       {"em": 40.2, "f1": 49.8, "avg_retrieval_rounds": 2.3, "avg_lm_calls": 2.5},
        "AdaptiveRAG": {"em": 44.1, "f1": 53.7, "avg_retrieval_rounds": 2.0, "avg_lm_calls": 2.2},
    }
}


def _print_table(dataset: str, methods: dict[str, dict], use_paper_baselines: bool) -> None:
    print(f"\n=== Results on {dataset} (n={next(iter(methods.values()), {}).get('n', '?')}) ===")
    header = (f"{'Method':<18} {'EM':>6} {'F1':>6} {'Contains%':>10} "
              f"{'Avg-Ret':>8} {'Avg-LM':>8} {'EStop%':>7} "
              f"{'DAG_Succ%':>10} {'Avg_Br':>7}")
    print(header)
    print("-" * len(header))

    def _row(label: str, m: dict) -> None:
        contains = m.get('contains', 0.0)
        estop = m.get('early_stop_rate', 0.0)
        dag_succ = m.get('dag_success_rate', 0.0)
        avg_br = m.get('avg_dag_branches', 0.0)
        print(f"{label:<18} {m['em']:>6.1f} {m['f1']:>6.1f} {contains:>9.1f}% "
              f"{m['avg_retrieval_rounds']:>8.1f} {m['avg_lm_calls']:>8.1f} {estop:>6.1f}% "
              f"{dag_succ:>9.1f}% {avg_br:>7.2f}")

    # Baselines — prefer loaded results, fall back to paper numbers
    paper = _ADARAGUE_PAPER.get(dataset, {})
    for bl in _BASELINE_ORDER:
        # Try to find loaded baseline result
        loaded_key = next((k for k in methods if bl.lower() in k.lower()
                           and "ablation" not in k and "cbet" not in k), None)
        if loaded_key:
            _row(bl, methods[loaded_key])
        elif use_paper_baselines and bl in paper:
            _row(bl + " *", paper[bl])

    # CBET results
    cbet_key = next((k for k in methods if k.startswith("cbet_")), None)
    if cbet_key:
        m = methods[cbet_key]
        _row("CBET (ours)", m)
        print()
        print(f"  avg_cs_at_stop={m['avg_cs_at_stop']:.3f}  "
              f"conflict_rate={m['conflict_detected_rate']:.1f}%  "
              f"override_rate={m['override_triggered_rate']:.1f}%  "
              f"noisy_evicted={m['noisy_branch_evicted_rate']:.1f}%  "
              f"dag_success={m.get('dag_success_rate',0):.1f}%  "
              f"avg_dag_br={m.get('avg_dag_branches',0):.2f}  "
              f"dag_fallback={m.get('dag_fallback_rate',0):.1f}%")

    print("  (* = AdaRAGUE paper numbers, not re-run)")


def _print_ablation_table(methods: dict[str, dict]) -> None:
    ablation_map = {
        "ablation_full":         "Full CBET",
        "ablation_no_cross":     "w/o Cross-Branch NLI",
        "ablation_no_override":  "w/o Epistemic Override",
        "ablation_entropy":      "w/ Entropy Only",
        "ablation_fixed":        "Fixed Rounds (k=3)",
    }
    ablations = {k: v for k, v in methods.items() if k.startswith("ablation_")}
    if not ablations:
        return
    print("\n=== Ablation Study (HotpotQA) ===")
    header = f"{'Variant':<26} {'EM':>6} {'F1':>6} {'Avg-Ret':>8} {'Avg-LM':>8}"
    print(header)
    print("-" * len(header))
    for key in ["ablation_full", "ablation_no_cross", "ablation_no_override",
                "ablation_entropy", "ablation_fixed"]:
        if key in ablations:
            label = ablation_map.get(key, key)
            m = ablations[key]
            print(f"{label:<26} {m['em']:>6.1f} {m['f1']:>6.1f} "
                  f"{m['avg_retrieval_rounds']:>8.1f} {m['avg_lm_calls']:>8.1f}")


# ── stratified analysis ───────────────────────────────────────────────────────

def stratified_analysis(results_path: str, dataset: str):
    """
    HotpotQA: stratify by difficulty level (easy/medium/hard)
      - Note: difficulty != hop count; use "question complexity" in paper

    MuSiQue: stratify by real hop count (2hop/3hop/4hop from qid prefix)

    Output format (HotpotQA example):
    ┌──────────────┬──────┬──────┬──────┬──────┐
    │    Method    │ All  │ easy │ med  │ hard │
    ├──────────────┼──────┼──────┼──────┼──────┤
    │ SingleRAG F1 │      │      │      │      │
    │ CBET F1      │      │      │      │      │
    │ Δ            │      │      │      │      │
    └──────────────┴──────┴──────┴──────┴──────┘
    """
    if not Path(results_path).exists():
        print(f"[WARN] {results_path} not found, skipping stratified analysis")
        return

    logs = _load(results_path)
    if not logs:
        return

    # Determine stratification key
    if dataset == "hotpotqa":
        # HotpotQA: difficulty from 'difficulty' field or infer from qid
        def get_stratum(log: dict) -> str:
            return log.get("difficulty", "unknown")
        strata_order = ["easy", "medium", "hard", "unknown"]
        stratum_labels = {"easy": "easy", "medium": "med", "hard": "hard", "unknown": "?"}
    elif dataset == "musique":
        # MuSiQue: hop count from qid prefix (2hop__/3hop__/4hop__)
        def get_stratum(log: dict) -> str:
            qid = log.get("qid", "")
            if qid.startswith("2hop__"):
                return "2hop"
            elif qid.startswith("3hop__"):
                return "3hop"
            elif qid.startswith("4hop__"):
                return "4hop"
            return "unknown"
        strata_order = ["2hop", "3hop", "4hop", "unknown"]
        stratum_labels = {"2hop": "2hop", "3hop": "3hop", "4hop": "4hop", "unknown": "?"}
    else:
        print(f"[INFO] Stratified analysis not implemented for {dataset}")
        return

    # Group logs by stratum
    strata: dict[str, list[dict]] = {s: [] for s in strata_order}
    for log in logs:
        s = get_stratum(log)
        if s in strata:
            strata[s].append(log)

    # Compute metrics per stratum
    print(f"\n=== Stratified Analysis ({dataset}) ===")
    header = f"{'Stratum':<10} {'N':>5} {'EM':>6} {'F1':>6} {'Avg-CS':>7} {'EStop%':>7}"
    print(header)
    print("-" * len(header))

    for s in strata_order:
        subset = strata[s]
        if not subset:
            continue
        n = len(subset)
        em = 100 * sum(log.get("em", 0) for log in subset) / n
        f1 = 100 * sum(log.get("f1", 0) for log in subset) / n
        cs = sum(log.get("final_cs", 0.0) for log in subset) / n
        estop = 100 * sum(1 for log in subset if log.get("early_stopped", False)) / n
        label = stratum_labels.get(s, s)
        print(f"{label:<10} {n:>5} {em:>6.1f} {f1:>6.1f} {cs:>7.3f} {estop:>6.1f}%")


# ── failure mode analysis ─────────────────────────────────────────────────────

def failure_mode_analysis(results_path: str):
    """
    Analyze CBET failure modes:

    Type A: "Confident but wrong" (high CS, EM=0)
      - Definition: CS >= 0.5 and EM = 0
      - Stats: proportion of early-stopped samples
      - Possible causes: distractor dominance, bridge entity semantic drift,
        internally consistent but incorrect reasoning chain

    Type B: "Correct but low confidence" (CS~0, EM=1)
      - Definition: CS < 0.1 and EM = 1
      - Stats: count
      - Possible causes: edge support fails to recognize semantically equivalent
        bridge entities

    Output format:
    === Failure Mode Analysis ===
    Type A (High CS, Wrong): X/500 (XX% of early stops)
      Sample cases (print 3 examples):
        - question / CBET answer / gold / CS score / edge_scores

    Type B (Low CS, Correct): X/500 (XX% of correct answers)
      Sample cases (print 3 examples):
        - question / CBET answer / gold / CS score / edge_scores
    """
    if not Path(results_path).exists():
        print(f"[WARN] {results_path} not found, skipping failure mode analysis")
        return

    logs = _load(results_path)
    if not logs:
        return

    n = len(logs)

    # Type A: high CS but wrong
    type_a = [log for log in logs if log.get("final_cs", 0.0) >= 0.5 and log.get("em", 0) == 0]
    early_stopped = [log for log in logs if log.get("early_stopped", False)]
    type_a_pct_of_estop = (len(type_a) / len(early_stopped) * 100) if early_stopped else 0.0

    # Type B: low CS but correct
    type_b = [log for log in logs if log.get("final_cs", 0.0) < 0.1 and log.get("em", 0) == 1]
    correct_answers = [log for log in logs if log.get("em", 0) == 1]
    type_b_pct_of_correct = (len(type_b) / len(correct_answers) * 100) if correct_answers else 0.0

    # Type C: DAG fallback samples — CBET degraded to flat IterativeRAG (single-node DAG)
    type_c = [log for log in logs if log.get("dag_fallback", False)]
    type_c_em = (100 * sum(1 for log in type_c if log.get("em", 0) == 1) / len(type_c)) if type_c else 0.0
    type_c_f1 = (100 * sum(_f1(log.get("answer", ""), log.get("gold_answer", "")) for log in type_c) / len(type_c)) if type_c else 0.0
    non_c = [log for log in logs if not log.get("dag_fallback", False)]
    non_c_em = (100 * sum(1 for log in non_c if log.get("em", 0) == 1) / len(non_c)) if non_c else 0.0
    non_c_f1 = (100 * sum(_f1(log.get("answer", ""), log.get("gold_answer", "")) for log in non_c) / len(non_c)) if non_c else 0.0

    print(f"\n=== Failure Mode Analysis ===")
    print(f"Type A (High CS, Wrong): {len(type_a)}/{n} ({type_a_pct_of_estop:.1f}% of early stops)")
    print(f"Type B (Low CS, Correct): {len(type_b)}/{n} ({type_b_pct_of_correct:.1f}% of correct answers)")
    print(f"Type C (DAG fallback): {len(type_c)}/{n} "
          f"({100*len(type_c)/n:.1f}% of total)")
    if type_c:
        print(f"  DAG fallback samples — EM={type_c_em:.1f}, F1={type_c_f1:.1f}")
        print(f"  Non-fallback samples — EM={non_c_em:.1f}, F1={non_c_f1:.1f}")

    # Print 3 examples of each type
    print(f"\nType A samples (up to 3):")
    for log in type_a[:3]:
        print(f"  QID: {log.get('qid', '?')}")
        print(f"    Answer: {log.get('answer', '')}")
        print(f"    Gold: {log.get('gold_answer', '')}")
        print(f"    CS: {log.get('final_cs', 0.0):.3f}")
        print(f"    Edge scores: {log.get('edge_scores', [])}")

    print(f"\nType B samples (up to 3):")
    for log in type_b[:3]:
        print(f"  QID: {log.get('qid', '?')}")
        print(f"    Answer: {log.get('answer', '')}")
        print(f"    Gold: {log.get('gold_answer', '')}")
        print(f"    CS: {log.get('final_cs', 0.0):.3f}")
        print(f"    Edge scores: {log.get('edge_scores', [])}")

    print(f"\nType C samples (up to 3):")
    for log in type_c[:3]:
        print(f"  QID: {log.get('qid', '?')}")
        print(f"    Query: {log.get('query', log.get('root_query', ''))[:80]}")
        print(f"    Answer: {log.get('answer', '')}  |  Gold: {log.get('gold_answer', '')}")
        print(f"    CS: {log.get('final_cs', 0.0):.3f}  |  EM: {log.get('em', 0)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",        default="experiments/results/")
    parser.add_argument("--dataset",            default=None,
                        choices=["hotpotqa", "musique", "2wikimultihopqa", None])
    parser.add_argument("--use_paper_baselines", action="store_true", default=True,
                        help="Show AdaRAGUE paper numbers when baseline JSONs are absent")
    parser.add_argument("--save_summary",       default=None,
                        help="Optional path to save aggregated metrics as JSON")
    parser.add_argument("--stratified",         action="store_true",
                        help="Run stratified analysis (difficulty/hop count)")
    parser.add_argument("--failure_modes",      action="store_true",
                        help="Run failure mode analysis")
    args = parser.parse_args()

    datasets = (
        [args.dataset] if args.dataset
        else ["hotpotqa", "musique", "2wikimultihopqa"]
    )

    all_metrics: dict[str, dict] = {}
    for ds in datasets:
        methods = _find_results(args.results_dir, ds)
        if methods:
            _print_table(ds, methods, args.use_paper_baselines)
            all_metrics[ds] = methods

            # Run stratified analysis if requested and CBET results exist
            if args.stratified:
                cbet_key = next((k for k in methods if k.startswith("cbet_")), None)
                if cbet_key:
                    cbet_path = str(Path(args.results_dir) / f"{cbet_key}.json")
                    stratified_analysis(cbet_path, ds)

            # Run failure mode analysis if requested and CBET results exist
            if args.failure_modes:
                cbet_key = next((k for k in methods if k.startswith("cbet_")), None)
                if cbet_key:
                    cbet_path = str(Path(args.results_dir) / f"{cbet_key}.json")
                    failure_mode_analysis(cbet_path)

    # Ablation table (dataset-agnostic keys)
    all_methods = _find_results(args.results_dir, dataset=None)
    _print_ablation_table(all_methods)

    if args.save_summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_summary)), exist_ok=True)
        with open(args.save_summary, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to {args.save_summary}")


if __name__ == "__main__":
    main()
