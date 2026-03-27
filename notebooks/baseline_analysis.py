"""
Baseline analysis: load results, compare methods, generate leaderboard.

Usage:
    uv run notebooks/baseline_analysis.py [--preset random_train] [--results-dir results]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_results(results_dir: Path, preset: str | None = None) -> list[dict]:
    """Load all result JSON files, optionally filtered by preset name."""
    results = []
    for path in sorted(results_dir.rglob("*.json")):
        parts = path.relative_to(results_dir).parts
        # Expected: method/preset/model/sample_S_seed_N.json
        if len(parts) < 3:
            continue
        method = parts[0]
        result_preset = parts[1]
        if preset and result_preset != preset:
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            data["_method"] = method
            data["_preset"] = result_preset
            data["_path"] = str(path)
            results.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Skipping {path}: {e}")
    return results


def build_leaderboard(results: list[dict]) -> pd.DataFrame:
    """Build a leaderboard comparing methods by final loss."""
    rows = []
    by_method = defaultdict(list)
    for r in results:
        by_method[r["_method"]].append(r)

    for method, runs in sorted(by_method.items()):
        losses = [r["best_loss"] for r in runs]
        match_rates = [r.get("match_rate", 0) or 0 for r in runs]
        flops = [r.get("total_flops", 0) for r in runs]

        rows.append({
            "method": method,
            "n_runs": len(runs),
            "mean_loss": sum(losses) / len(losses),
            "std_loss": pd.Series(losses).std(),
            "best_loss": min(losses),
            "worst_loss": max(losses),
            "mean_match_rate": sum(match_rates) / len(match_rates),
            "mean_flops": sum(flops) / len(flops),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_loss")
    return df


def main():
    parser = argparse.ArgumentParser(description="Claudini baseline analysis")
    parser.add_argument("--preset", default=None, help="Filter by preset name (e.g. random_train)")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run benchmarks first, or download precomputed results from GitHub releases.")
        return

    results = load_results(results_dir, args.preset)
    if not results:
        print(f"No results found in {results_dir}" + (f" for preset '{args.preset}'" if args.preset else ""))
        return

    print(f"Loaded {len(results)} result files")
    if args.preset:
        print(f"Preset: {args.preset}")
    print()

    df = build_leaderboard(results)
    print("=" * 90)
    print("LEADERBOARD (sorted by mean loss, lower is better)")
    print("=" * 90)
    print(df.to_string(index=False, float_format="%.4f"))
    print()

    # Per-sample breakdown for top methods
    top_methods = df.head(5)["method"].tolist()
    print(f"Per-sample breakdown (top {len(top_methods)} methods):")
    print("-" * 60)
    for method in top_methods:
        runs = [r for r in results if r["_method"] == method]
        samples = sorted(set(r.get("sample_id", "?") for r in runs))
        losses_by_sample = {}
        for r in runs:
            sid = r.get("sample_id", "?")
            losses_by_sample.setdefault(sid, []).append(r["best_loss"])
        parts = [f"s{s}={sum(v)/len(v):.4f}" for s, v in sorted(losses_by_sample.items())]
        print(f"  {method:30s} {' '.join(parts)}")


if __name__ == "__main__":
    main()
