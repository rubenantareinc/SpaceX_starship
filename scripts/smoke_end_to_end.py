"""Run a lightweight end-to-end pipeline and emit output paths."""

import subprocess
import sys


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    run(
        [
            sys.executable,
            "-m",
            "src.ingest.build_incidents",
            "--raw-dir",
            "data/raw_text",
            "--sources",
            "data/sources.csv",
            "--out",
            "data/processed/incidents.jsonl",
            "--labels-from",
            "data/processed/incidents.jsonl",
        ]
    )
    run([sys.executable, "-m", "src.eval.split", "--data", "data/processed/incidents.jsonl"])
    run(
        [
            sys.executable,
            "-m",
            "src.baselines.keyword_baseline",
            "--data",
            "data/processed/incidents.jsonl",
            "--split",
            "outputs/split.json",
            "--out",
            "outputs/keyword_preds.jsonl",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.baselines.tfidf_baseline",
            "--data",
            "data/processed/incidents.jsonl",
            "--split",
            "outputs/split.json",
            "--out",
            "outputs/tfidf_preds.jsonl",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.eval.evaluate",
            "--gold",
            "data/processed/incidents.jsonl",
            "--pred",
            "outputs/keyword_preds.jsonl",
            "--split",
            "outputs/split.json",
            "--out",
            "outputs/keyword_metrics.json",
            "--md-out",
            "outputs/keyword_metrics.md",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.eval.evaluate",
            "--gold",
            "data/processed/incidents.jsonl",
            "--pred",
            "outputs/tfidf_preds.jsonl",
            "--split",
            "outputs/split.json",
            "--out",
            "outputs/tfidf_metrics.json",
            "--md-out",
            "outputs/tfidf_metrics.md",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.eval.evidence_eval",
            "--gold",
            "data/processed/incidents.jsonl",
            "--pred",
            "outputs/keyword_preds.jsonl",
            "--split",
            "outputs/split.json",
            "--out",
            "outputs/evidence_metrics.json",
            "--md-out",
            "outputs/evidence_metrics.md",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.eval.dataset_stats",
            "--data",
            "data/processed/incidents.jsonl",
            "--stats-out",
            "outputs/dataset_stats.json",
            "--labels-out",
            "outputs/label_distribution.json",
        ]
    )

    print("\nOutputs written:")
    print("- outputs/split.json")
    print("- outputs/keyword_preds.jsonl")
    print("- outputs/tfidf_preds.jsonl")
    print("- outputs/keyword_metrics.json")
    print("- outputs/keyword_metrics.md")
    print("- outputs/tfidf_metrics.json")
    print("- outputs/tfidf_metrics.md")
    print("- outputs/evidence_metrics.json")
    print("- outputs/evidence_metrics.md")
    print("- outputs/dataset_stats.json")
    print("- outputs/label_distribution.json")


if __name__ == "__main__":
    main()
