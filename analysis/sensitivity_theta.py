"""θ hyperparameter sensitivity analysis.

Runs CBET on HotpotQA with θ ∈ [0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
plots θ vs F1 and θ vs avg retrieval rounds on a dual-axis line chart,
saves to experiments/results/theta_sensitivity.png.

Usage:
    python analysis/sensitivity_theta.py \
        --model_path ./models/ \
        --n_samples 100 \
        --output experiments/results/theta_sensitivity.png
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

THETAS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def run_theta(theta: float, args) -> dict:
    """Run CBET for one theta value, return aggregated metrics."""
    from src.cbet_controller import CBETController, CBETConfig
    from src.llm_client import build_client
    from src.nli_scorer import NLIScorer
    from src.parametric_probe import ParametricProbe
    from src.retriever import ElasticRetriever
    from src.data_adapter import load_dataset

    model_path = os.path.join(args.model_path, "Qwen2.5-7B-Instruct-AWQ")
    nli_path   = os.path.join(args.model_path, "nli-deberta-v3-base")

    llm      = build_client("awq", model_path)
    nli      = NLIScorer(model_path=nli_path, theta=theta)
    probe    = ParametricProbe(llm)
    retriever = ElasticRetriever(index_name="hotpotqa")
    config   = CBETConfig(theta=theta, tau=0.5, max_iterations=5)
    ctrl     = CBETController(llm, retriever, nli, probe, config)

    questions = load_dataset("hotpotqa", n_samples=args.n_samples)
    logs = []
    for q in questions:
        try:
            r = ctrl.solve(q)
            logs.append(r.log)
        except Exception as e:
            print(f"  [WARN] qid={q.qid}: {e}")

    f1   = sum(l["f1"] for l in logs) / len(logs) if logs else 0.0
    iters = sum(l["iterations"] for l in logs) / len(logs) if logs else 0.0
    return {"theta": theta, "f1": f1, "avg_iterations": iters, "n": len(logs)}


def plot(records: list[dict], output: str) -> None:
    import matplotlib.pyplot as plt

    thetas = [r["theta"] for r in records]
    f1s    = [r["f1"] for r in records]
    iters  = [r["avg_iterations"] for r in records]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(thetas, f1s,   "b-o", label="F1")
    ax2.plot(thetas, iters, "r--s", label="Avg Retrieval Rounds")

    ax1.set_xlabel("θ (completeness threshold)")
    ax1.set_ylabel("Token F1", color="b")
    ax2.set_ylabel("Avg Retrieval Rounds", color="r")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("CBET: θ Sensitivity on HotpotQA")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./models/")
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--output",     default="experiments/results/theta_sensitivity.png")
    parser.add_argument("--results_json", default="experiments/results/theta_sensitivity.json")
    args = parser.parse_args()

    records = []
    for theta in THETAS:
        print(f"\n=== θ={theta} ===")
        rec = run_theta(theta, args)
        records.append(rec)
        print(f"  F1={rec['f1']:.4f}  avg_iters={rec['avg_iterations']:.2f}  n={rec['n']}")

    os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
    with open(args.results_json, "w") as f:
        json.dump(records, f, indent=2)

    plot(records, args.output)


if __name__ == "__main__":
    main()
