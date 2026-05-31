"""Generate paper-ready tables from experiment results.

Reads from:
  - experiments/results/exp1_main.json       → Table 1 (main comparison)
  - experiments/results/exp1_main.json       → Table 2 (complexity stratification)
  - experiments/results/exp2_ablation.json   → Table 3 (ablation study)
  - experiments/results/exp3_theta.json      → Figure data (θ sensitivity)

Usage:
    uv run python analysis/generate_paper_tables.py \
        --results_dir experiments/results/

    uv run python analysis/generate_paper_tables.py \
        --results_dir experiments/results/ --output paper_tables.md
"""
import argparse
import json
import os
from pathlib import Path


# ── metric helpers ────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


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
    rec = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def _aggregate(entries: list[dict]) -> dict:
    """Aggregate per-question results into summary metrics.

    Handles both exp1 format (answer/gold_answer) and log format (em/f1 fields).
    """
    n = len(entries)
    if n == 0:
        return {}

    em_scores = [e.get("em", _em(e.get("answer", ""), e.get("gold_answer", "")))
                 for e in entries]
    f1_scores = [e.get("f1", _f1(e.get("answer", ""), e.get("gold_answer", "")))
                 for e in entries]
    iters = [e.get("retrieval_rounds", e.get("iterations", 1)) for e in entries]
    cs_scores = [e.get("final_cs", 0.0) for e in entries]
    lm_calls = [e.get("lm_calls", e.get("total_lm_calls", 0)) for e in entries]

    # Contains%: gold answer is substring of predicted answer
    contains_count = 0
    for e in entries:
        pred = e.get("answer", "").strip().lower()
        gold = e.get("gold_answer", "").strip().lower()
        if gold and gold in pred:
            contains_count += 1

    # Early stop (only in exp3 theta results)
    early_stop = [e.get("early_stopped", False) for e in entries]

    return {
        "n": n,
        "em": 100 * sum(em_scores) / n,
        "f1": 100 * sum(f1_scores) / n,
        "contains": 100 * contains_count / n,
        "avg_ret": sum(iters) / n,
        "avg_lm": sum(lm_calls) / n,
        "avg_cs": sum(cs_scores) / n,
        "estop_pct": 100 * sum(early_stop) / n if any(early_stop) else 0.0,
    }


def _group_by(entries: list[dict], key: str) -> dict[str, list[dict]]:
    """Group entries by a key field."""
    groups: dict[str, list[dict]] = {}
    for e in entries:
        k = e.get(key, "unknown")
        if k not in groups:
            groups[k] = []
        groups[k].append(e)
    return groups


# ── Table 1: Main comparison ──────────────────────────────────────────────────

def generate_table1_main(results_dir: str, dataset: str) -> str:
    """Table 1: Main comparison across methods (NoRAG, SingleRAG, IterativeRAG, CBET)."""
    lines = [f"## Table 1: Main Results on {dataset}\n"]

    exp1_path = os.path.join(results_dir, "exp1_main.json")
    if not os.path.exists(exp1_path):
        return lines[0] + "\n*No exp1_main.json found*\n"

    entries = _load(exp1_path)
    # Filter by dataset and exclude errors
    ds_entries = [e for e in entries
                  if e.get("dataset") == dataset and "error" not in e]

    if not ds_entries:
        return lines[0] + f"\n*No results for {dataset}*\n"

    by_method = _group_by(ds_entries, "method")

    # Header
    lines.append("| Method | EM | F1 | Contains% | Avg-Ret | Avg-LM | EStop% |")
    lines.append("|--------|----|----|-----------|---------|--------|--------|")

    # Method order: baselines first, then CBET
    method_order = ["NoRAG", "SingleRAG", "IterativeRAG", "CBET"]
    for method in method_order:
        if method not in by_method:
            continue
        m = _aggregate(by_method[method])
        label = "CBET (ours)" if method == "CBET" else method
        lines.append(
            f"| {label} | {m['em']:.1f} | {m['f1']:.1f} | {m['contains']:.1f}% | "
            f"{m['avg_ret']:.2f} | {m['avg_lm']:.1f} | {m['estop_pct']:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


# ── Table 2: Stratification ───────────────────────────────────────────────────

def generate_table2_stratification(results_dir: str, dataset: str) -> str:
    """Table 2: Complexity stratification (CBET only).

    HotpotQA: by difficulty level (easy/medium/hard)
    MuSiQue: by hop count (2hop/3hop/4hop from qid prefix)
    """
    lines = [f"## Table 2: Stratified Analysis on {dataset}\n"]

    exp1_path = os.path.join(results_dir, "exp1_main.json")
    if not os.path.exists(exp1_path):
        return lines[0] + "\n*No exp1_main.json found*\n"

    entries = _load(exp1_path)
    # Only CBET entries for this dataset
    cbet_entries = [e for e in entries
                    if e.get("method") == "CBET"
                    and e.get("dataset") == dataset
                    and "error" not in e]

    if not cbet_entries:
        return lines[0] + f"\n*No CBET results for {dataset}*\n"

    # Determine stratification
    if dataset == "hotpotqa":
        def get_stratum(e: dict) -> str:
            return e.get("difficulty", "unknown")
        strata_order = ["easy", "medium", "hard", "unknown"]
        stratum_labels = {"easy": "Easy", "medium": "Medium",
                          "hard": "Hard", "unknown": "All"}
    elif dataset == "musique":
        def get_stratum(e: dict) -> str:
            qid = e.get("qid", "")
            if qid.startswith("2hop__"):
                return "2hop"
            elif qid.startswith("3hop__") or qid.startswith("3hop1__"):
                return "3hop"
            elif qid.startswith("4hop__"):
                return "4hop"
            return "unknown"
        strata_order = ["2hop", "3hop", "4hop", "unknown"]
        stratum_labels = {"2hop": "2-hop", "3hop": "3-hop",
                          "4hop": "4-hop", "unknown": "Other"}
    else:
        return lines[0] + f"\n*Stratification not implemented for {dataset}*\n"

    # Group by stratum
    strata: dict[str, list[dict]] = {s: [] for s in strata_order}
    for e in cbet_entries:
        s = get_stratum(e)
        if s in strata:
            strata[s].append(e)

    # Header
    lines.append("| Stratum | N | EM | F1 | Avg-CS | EStop% |")
    lines.append("|---------|---|----|----|--------|--------|")

    # Add "All" row first
    all_m = _aggregate(cbet_entries)
    lines.append(
        f"| **All** | {all_m['n']} | {all_m['em']:.1f} | {all_m['f1']:.1f} | "
        f"{all_m['avg_cs']:.3f} | {all_m['estop_pct']:.1f}% |"
    )

    for s in strata_order:
        subset = strata[s]
        if not subset:
            continue
        m = _aggregate(subset)
        label = stratum_labels.get(s, s)
        lines.append(
            f"| {label} | {m['n']} | {m['em']:.1f} | {m['f1']:.1f} | "
            f"{m['avg_cs']:.3f} | {m['estop_pct']:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


# ── Table 3: Ablation ────────────────────────────────────────────────────────

def generate_table3_ablation(results_dir: str) -> str:
    """Table 3: Ablation study (5 variants)."""
    lines = ["## Table 3: Ablation Study\n"]

    exp2_path = os.path.join(results_dir, "exp2_ablation.json")
    if not os.path.exists(exp2_path):
        return lines[0] + "\n*No exp2_ablation.json found*\n"

    entries = _load(exp2_path)
    entries = [e for e in entries if "error" not in e]

    if not entries:
        return lines[0] + "\n*No ablation results found*\n"

    by_variant = _group_by(entries, "variant")

    variant_labels = {
        "full": "Full CBET",
        "no_cross_branch": "w/o Cross-Branch NLI",
        "no_override": "w/o Epistemic Override",
        "no_early_stop": "w/o Early Stop",
        "single_branch": "Single Branch (k=1)",
    }
    variant_order = ["full", "no_cross_branch", "no_override",
                     "no_early_stop", "single_branch"]

    # Header
    lines.append("| Variant | N | EM | F1 | Contains% | Avg-Ret | Avg-LM |")
    lines.append("|---------|---|----|----|-----------|---------|--------|")

    for v in variant_order:
        if v not in by_variant:
            continue
        m = _aggregate(by_variant[v])
        label = variant_labels.get(v, v)
        lines.append(
            f"| {label} | {m['n']} | {m['em']:.1f} | {m['f1']:.1f} | "
            f"{m['contains']:.1f}% | {m['avg_ret']:.2f} | {m['avg_lm']:.1f} |"
        )

    lines.append("")
    return "\n".join(lines)


# ── Figure data: θ sensitivity ───────────────────────────────────────────────

def generate_figure_theta(results_dir: str) -> str:
    """Figure data: θ sensitivity (F1 and EarlyStop% vs θ)."""
    lines = ["## Figure Data: Theta Sensitivity\n"]

    exp3_path = os.path.join(results_dir, "exp3_theta.json")
    if not os.path.exists(exp3_path):
        return lines[0] + "\n*No exp3_theta.json found*\n"

    entries = _load(exp3_path)
    entries = [e for e in entries if "error" not in e]

    if not entries:
        return lines[0] + "\n*No theta results found*\n"

    by_theta = _group_by(entries, "theta")

    # Sort by theta value
    theta_values = sorted(by_theta.keys(), key=float)

    # Header
    lines.append("| Theta | N | EM | F1 | EarlyStop% | Avg-Ret | Avg-CS |")
    lines.append("|-------|---|----|----|------------|---------|--------|")

    for theta in theta_values:
        m = _aggregate(by_theta[theta])
        lines.append(
            f"| {float(theta):.2f} | {m['n']} | {m['em']:.1f} | {m['f1']:.1f} | "
            f"{m['estop_pct']:.1f}% | {m['avg_ret']:.2f} | {m['avg_cs']:.3f} |"
        )

    lines.append("")
    lines.append("*Plot: X-axis = Theta, Y1 (left) = F1, Y2 (right) = EarlyStop%*")
    lines.append("")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper tables from experiment results")
    parser.add_argument("--results_dir", default="experiments/results/",
                        help="Directory containing experiment result JSONs")
    parser.add_argument("--output", default="experiments/results/paper_tables.md",
                        help="Output path for markdown tables")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "musique"],
                        help="Datasets to include in Table 1 and Table 2")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return

    output_lines = ["# Paper Tables\n", "*Auto-generated from experiment results*\n"]

    # Table 1: Main comparison (per dataset)
    for ds in args.datasets:
        output_lines.append(generate_table1_main(str(results_dir), ds))

    # Table 2: Stratification (per dataset)
    for ds in args.datasets:
        output_lines.append(generate_table2_stratification(str(results_dir), ds))

    # Table 3: Ablation
    output_lines.append(generate_table3_ablation(str(results_dir)))

    # Figure data: Theta sensitivity
    output_lines.append(generate_figure_theta(str(results_dir)))

    # Write output
    output_text = "\n".join(output_lines)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Paper tables generated: {args.output}")
    print(f"  Table 1 (main comparison): {len(args.datasets)} datasets")
    print(f"  Table 2 (stratification): {len(args.datasets)} datasets")
    print(f"  Table 3 (ablation: 5 variants)")
    print(f"  Figure  (θ sensitivity)")


if __name__ == "__main__":
    main()
