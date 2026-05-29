"""Generate paper-ready tables from experiment results.

Usage:
    python analysis/generate_paper_tables.py --results_dir experiments/results/
    python analysis/generate_paper_tables.py --results_dir experiments/results/ --output paper_tables.md

This script:
  - Scans results_dir for JSON files matching experiment patterns
  - Generates Table 1 (main comparison), Table 2 (complexity stratification),
    Table 3 (ablation), and Figure data (theta sensitivity)
  - Outputs formatted markdown tables without directional claims
"""
import argparse
import json
from pathlib import Path


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
    rec  = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def _aggregate(logs: list[dict]) -> dict:
    n = len(logs)
    if n == 0:
        return {}

    em_scores = [log.get("em", _em(log.get("answer",""), log.get("gold_answer",""))) for log in logs]
    f1_scores = [log.get("f1", _f1(log.get("answer",""), log.get("gold_answer",""))) for log in logs]
    iters = [log.get("iterations", 1) for log in logs]
    cs_scores = [log.get("final_cs", 0.0) for log in logs]
    lm_calls = [log.get("total_lm_calls", 1 + log.get("iterations",1) * (log.get("dag_size",1) * 3) + 1) for log in logs]
    early_stop = [log.get("early_stopped", False) for log in logs]

    # Contains%
    contains_count = 0
    for log in logs:
        pred = log.get("answer", "").strip().lower()
        gold = log.get("gold_answer", "").strip().lower()
        if gold and gold in pred:
            contains_count += 1

    return {
        "n": n,
        "em": 100 * sum(em_scores) / n,
        "f1": 100 * sum(f1_scores) / n,
        "contains": 100 * contains_count / n,
        "avg_ret": sum(iters) / n,
        "avg_lm": sum(lm_calls) / n,
        "avg_cs": sum(cs_scores) / n,
        "estop_pct": 100 * sum(early_stop) / n,
    }


def generate_table1_main(results_dir: str, dataset: str) -> str:
    """Table 1: Main comparison across methods."""
    lines = [f"## Table 1: Main Results on {dataset}\n"]

    # Find all result files for this dataset
    methods = {}
    for p in sorted(Path(results_dir).glob("*.json")):
        name = p.stem
        if dataset not in name or "ablation" in name or "theta" in name:
            continue
        try:
            logs = _load(str(p))
            metrics = _aggregate(logs)
            if metrics:
                methods[name] = metrics
        except Exception:
            continue

    if not methods:
        return lines[0] + "\n*No results found*\n"

    # Header
    lines.append("| Method | EM | F1 | Contains% | Avg-Ret | Avg-LM | EStop% |")
    lines.append("|--------|----|----|-----------|---------|--------|--------|")

    # Sort methods: baselines first, then CBET
    baseline_names = ["norag", "singlerag", "iterativerag", "dragin", "seakr", "flare", "adaptive"]
    sorted_methods = []
    for name in sorted(methods.keys()):
        if any(bn in name.lower() for bn in baseline_names):
            sorted_methods.append((name, methods[name]))
    for name in sorted(methods.keys()):
        if name.startswith("cbet_"):
            sorted_methods.append((name, methods[name]))

    for name, m in sorted_methods:
        label = name.replace("_", " ").title()
        if name.startswith("cbet_"):
            label = "CBET (ours)"
        lines.append(f"| {label} | {m['em']:.1f} | {m['f1']:.1f} | {m['contains']:.1f}% | "
                     f"{m['avg_ret']:.2f} | {m['avg_lm']:.1f} | {m['estop_pct']:.1f}% |")

    lines.append("")
    return "\n".join(lines)


def generate_table2_stratification(results_dir: str, dataset: str) -> str:
    """Table 2: Complexity stratification."""
    lines = [f"## Table 2: Stratified Analysis on {dataset}\n"]

    # Find CBET results
    cbet_path = None
    for p in Path(results_dir).glob("cbet_*.json"):
        if dataset in p.stem and "ablation" not in p.stem:
            cbet_path = p
            break

    if not cbet_path or not cbet_path.exists():
        return lines[0] + "\n*No CBET results found*\n"

    logs = _load(str(cbet_path))
    if not logs:
        return lines[0] + "\n*No logs found*\n"

    # Determine stratification
    if dataset == "hotpotqa":
        def get_stratum(log: dict) -> str:
            return log.get("difficulty", "unknown")
        strata_order = ["easy", "medium", "hard", "unknown"]
        stratum_labels = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "unknown": "Unknown"}
    elif dataset == "musique":
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
        stratum_labels = {"2hop": "2-hop", "3hop": "3-hop", "4hop": "4-hop", "unknown": "Unknown"}
    else:
        return lines[0] + f"\n*Stratification not implemented for {dataset}*\n"

    # Group logs
    strata = {s: [] for s in strata_order}
    for log in logs:
        s = get_stratum(log)
        if s in strata:
            strata[s].append(log)

    # Header
    lines.append("| Stratum | N | EM | F1 | Avg-CS | EStop% |")
    lines.append("|---------|---|----|----|--------|--------|")

    for s in strata_order:
        subset = strata[s]
        if not subset:
            continue
        m = _aggregate(subset)
        label = stratum_labels.get(s, s)
        lines.append(f"| {label} | {m['n']} | {m['em']:.1f} | {m['f1']:.1f} | "
                     f"{m['avg_cs']:.3f} | {m['estop_pct']:.1f}% |")

    lines.append("")
    return "\n".join(lines)


def generate_table3_ablation(results_dir: str) -> str:
    """Table 3: Ablation study."""
    lines = ["## Table 3: Ablation Study\n"]

    ablation_map = {
        "ablation_full": "Full CBET",
        "ablation_no_cross": "w/o Cross-Branch NLI",
        "ablation_no_override": "w/o Epistemic Override",
        "ablation_entropy": "w/ Entropy Only",
        "ablation_fixed": "Fixed Rounds (k=3)",
    }

    methods = {}
    for p in Path(results_dir).glob("ablation_*.json"):
        try:
            logs = _load(str(p))
            metrics = _aggregate(logs)
            if metrics:
                methods[p.stem] = metrics
        except Exception:
            continue

    if not methods:
        return lines[0] + "\n*No ablation results found*\n"

    # Header
    lines.append("| Variant | EM | F1 | Avg-Ret | Avg-LM |")
    lines.append("|---------|----|----|---------|--------|")

    for key in ["ablation_full", "ablation_no_cross", "ablation_no_override",
                "ablation_entropy", "ablation_fixed"]:
        if key in methods:
            m = methods[key]
            label = ablation_map.get(key, key)
            lines.append(f"| {label} | {m['em']:.1f} | {m['f1']:.1f} | "
                         f"{m['avg_ret']:.2f} | {m['avg_lm']:.1f} |")

    lines.append("")
    return "\n".join(lines)


def generate_figure_theta(results_dir: str) -> str:
    """Figure data: theta sensitivity."""
    lines = ["## Figure Data: Theta Sensitivity\n"]

    # Find theta experiment results
    theta_results = []
    for p in sorted(Path(results_dir).glob("theta_*.json")):
        try:
            # Extract theta value from filename (theta_0.50.json)
            theta_str = p.stem.replace("theta_", "")
            theta_val = float(theta_str)
            logs = _load(str(p))
            metrics = _aggregate(logs)
            if metrics:
                theta_results.append((theta_val, metrics))
        except Exception:
            continue

    if not theta_results:
        return lines[0] + "\n*No theta sensitivity results found*\n"

    # Sort by theta
    theta_results.sort(key=lambda x: x[0])

    # Header
    lines.append("| Theta | EM | F1 | Avg-Ret | Avg-CS | EStop% |")
    lines.append("|-------|----|----|---------|--------|--------|")

    for theta, m in theta_results:
        lines.append(f"| {theta:.2f} | {m['em']:.1f} | {m['f1']:.1f} | "
                     f"{m['avg_ret']:.2f} | {m['avg_cs']:.3f} | {m['estop_pct']:.1f}% |")

    lines.append("")
    lines.append("*Plot: X-axis = Theta, Y-axis (left) = F1, Y-axis (right) = Avg-Ret*")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables from experiment results")
    parser.add_argument("--results_dir", default="experiments/results/",
                        help="Directory containing experiment result JSONs")
    parser.add_argument("--output", default="experiments/results/paper_tables.md",
                        help="Output path for markdown tables")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "musique"],
                        help="Datasets to include in tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist")
        return

    # Generate all tables
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
    print(f"  - Table 1 (main comparison): {len(args.datasets)} datasets")
    print(f"  - Table 2 (stratification): {len(args.datasets)} datasets")
    print(f"  - Table 3 (ablation)")
    print(f"  - Figure data (theta sensitivity)")


if __name__ == "__main__":
    main()
