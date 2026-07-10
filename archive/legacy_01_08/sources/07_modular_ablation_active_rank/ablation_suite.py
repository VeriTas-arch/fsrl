from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

KEYS = [
    "overall_accuracy",
    "learned_pairs_accuracy",
    "nonlearned_pairs_accuracy",
    "consistent_error_subjects_80pct_ratio",
    "consistent_error_subjects_100pct_ratio",
    "mean_self_consistency_from_majority_choices",
    "mean_circular_triads_from_majority_choices",
    "mean_inter_subject_kendall_tau",
    "correct_ranking_subjects",
    "self_consistent_incorrect_subjects",
    "self_inconsistent_subjects",
    "mean_hebb_gate",
    "mean_edge_strength",
    "edge_recon_loss",
]

HUMAN_ANCHORS = {
    "overall_accuracy": 0.84,
    "learned_pairs_accuracy": 0.92,
    "nonlearned_pairs_accuracy": 0.81,
    "consistent_error_subjects_80pct_ratio": 0.91,
    "consistent_error_subjects_100pct_ratio": 0.78,
    "mean_self_consistency_from_majority_choices": 1.00,
    "mean_inter_subject_kendall_tau": 0.55,
}


def run_cmd(cmd: list[str], timeout_s: int, log_path: Path) -> tuple[bool, float]:
    t0 = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        try:
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout_s)
            return p.returncode == 0, time.time() - t0
        except subprocess.TimeoutExpired:
            f.write(f"\n[TIMEOUT] command exceeded {timeout_s}s\n")
            return False, time.time() - t0


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "meta_plastic_paper_task_eval_summary.json"
    if not p.exists():
        p2 = run_dir / "meta_plastic_best_sweep_paper_task_eval_summary.json"
        if p2.exists():
            p = p2
    if not p.exists():
        return {"status": "missing_summary"}
    with p.open("r", encoding="utf-8") as f:
        s = json.load(f)
    return s


def row_from_summary(name: str, status: str, elapsed: float, summary: dict) -> dict:
    row = {"variant": name, "status": status, "elapsed_sec": round(elapsed, 3)}
    sig = summary.get("ablation_signature", {}) if isinstance(summary, dict) else {}
    for k, v in sig.items():
        row[k] = v
    for k in KEYS:
        row[k] = summary.get(k, "") if isinstance(summary, dict) else ""
    row["beta_bimodal_pairs"] = (summary.get("beta_pair_category_counts", {}) or {}).get("bimodal", "") if isinstance(summary, dict) else ""
    return row


def passfail(rows: list[dict]) -> list[str]:
    by = {r["variant"]: r for r in rows}
    lines: list[str] = []
    base = by.get("distance_rnn_reliability")
    no_rnn = by.get("distance_no_rnn_ablation")
    raw = by.get("raw_observation_rnn_reliability")
    raw_no = by.get("raw_observation_no_rnn_ablation")
    posterior_mean = by.get("distance_posterior_mean_readout")

    def f(row: dict, key: str) -> float | None:
        try:
            return float(row.get(key, ""))
        except Exception:
            return None

    if base and no_rnn:
        drop = (f(base, "overall_accuracy") or 0.0) - (f(no_rnn, "overall_accuracy") or 0.0)
        lines.append(f"- RNN necessity expectation: no-RNN should reduce accuracy by >=0.15. Observed drop = {drop:.3f} -> {'PASS' if drop >= 0.15 else 'CHECK'}.")
    if base:
        learned = f(base, "learned_pairs_accuracy") or 0.0
        nonlearned = f(base, "nonlearned_pairs_accuracy") or 0.0
        sc = f(base, "mean_self_consistency_from_majority_choices") or 0.0
        lines.append(f"- Behavioral-shape expectation: learned >= non-learned and self-consistency high. Observed learned/non-learned = {learned:.3f}/{nonlearned:.3f}, self-consistency = {sc:.3f} -> {'PASS' if learned >= nonlearned and sc >= 0.95 else 'CHECK'}.")
        overall = f(base, "overall_accuracy") or 0.0
        lines.append(f"- Human-fit expectation: full training should move overall accuracy toward ~{HUMAN_ANCHORS['overall_accuracy']:.2f}; observed = {overall:.3f}. Smoke runs may be lower.")
    if posterior_mean:
        sc_pm = f(posterior_mean, "mean_self_consistency_from_majority_choices") or 0.0
        lines.append(f"- Commitment/readout expectation: posterior-mean readout should reduce transitive self-consistency. Observed self-consistency = {sc_pm:.3f} -> {'PASS' if sc_pm < 0.95 else 'CHECK'}.")
    if raw and raw_no:
        raw_drop = (f(raw, "overall_accuracy") or 0.0) - (f(raw_no, "overall_accuracy") or 0.0)
        lines.append(f"- Raw-input expectation: raw bars are harder and may need longer training; no-RNN should not outperform trained RNN by much. Observed raw RNN/no-RNN = {(f(raw, 'overall_accuracy') or 0.0):.3f}/{(f(raw_no, 'overall_accuracy') or 0.0):.3f}, drop = {raw_drop:.3f} -> {'PASS' if raw_drop >= 0.05 else 'CHECK'}.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="ablation_runs")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--big-timeout", type=int, default=1200)
    ap.add_argument("--smoke", action="store_true", help="Use small settings for quick CI/smoke testing.")
    args = ap.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    script = str(Path(__file__).with_name("meta_plastic_rank_rnn.py"))

    if args.smoke:
        train_common = ["--nbiter", "30", "--batch-size", "4", "--hidden-size", "16", "--item-dim", "8", "--subject-dim", "6", "--eval-subjects", "24", "--eval-repetitions", "5", "--print-every", "10", "--save-every", "0"]
    else:
        train_common = ["--nbiter", "120", "--batch-size", "8", "--hidden-size", "32", "--item-dim", "16", "--subject-dim", "10", "--eval-subjects", "77", "--eval-repetitions", "10", "--print-every", "30", "--save-every", "30"]
    eval_common = ["--eval-only"] + train_common[2:]

    variants: list[tuple[str, list[str], int]] = []
    dist_dir = root / "distance_rnn_reliability"
    variants.append(("distance_rnn_reliability", [py, script, *train_common, "--reliability-mode", "rnn", "--relation-encoding-mode", "rnn", "--observation-mode", "distance", "--output-dir", str(dist_dir)], args.timeout if args.smoke else args.big_timeout))
    ckpt = dist_dir / "meta_plastic_rank_rnn.pt"
    variants.append(("distance_no_rnn_ablation", [py, script, *eval_common, "--load-checkpoint", str(ckpt), "--ablate-rnn", "--reliability-mode", "rnn", "--relation-encoding-mode", "rnn", "--observation-mode", "distance", "--output-dir", str(root / "distance_no_rnn_ablation")], args.timeout))
    variants.append(("distance_no_plasticity_ablation", [py, script, *eval_common, "--load-checkpoint", str(ckpt), "--ablate-plasticity", "--reliability-mode", "rnn", "--relation-encoding-mode", "rnn", "--observation-mode", "distance", "--output-dir", str(root / "distance_no_plasticity_ablation")], args.timeout))
    variants.append(("distance_constant_reliability_ablation", [py, script, *eval_common, "--load-checkpoint", str(ckpt), "--reliability-mode", "constant", "--relation-encoding-mode", "rnn", "--observation-mode", "distance", "--output-dir", str(root / "distance_constant_reliability_ablation")], args.timeout))
    variants.append(("distance_posterior_mean_readout", [py, script, *eval_common, "--load-checkpoint", str(ckpt), "--rank-readout", "posterior_mean", "--reliability-mode", "rnn", "--relation-encoding-mode", "rnn", "--observation-mode", "distance", "--output-dir", str(root / "distance_posterior_mean_readout")], args.timeout))

    raw_dir = root / "raw_observation_rnn_reliability"
    variants.append(("raw_observation_rnn_reliability", [py, script, *train_common, "--reliability-mode", "rnn", "--relation-encoding-mode", "rnn", "--observation-mode", "raw_bars", "--output-dir", str(raw_dir)], args.timeout if args.smoke else args.big_timeout))
    raw_ckpt = raw_dir / "meta_plastic_rank_rnn.pt"
    variants.append(("raw_observation_no_rnn_ablation", [py, script, *eval_common, "--load-checkpoint", str(raw_ckpt), "--ablate-rnn", "--reliability-mode", "rnn", "--relation-encoding-mode", "rnn", "--observation-mode", "raw_bars", "--output-dir", str(root / "raw_observation_no_rnn_ablation")], args.timeout))

    rows: list[dict] = []
    for name, cmd, timeout_s in variants:
        ok, elapsed = run_cmd(cmd, timeout_s, root / f"{name}.log")
        summary = load_summary(root / name)
        rows.append(row_from_summary(name, "ok" if ok else "failed_or_timeout", elapsed, summary))

    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with (root / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    with (root / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    lines = [
        "# Ablation expectation report",
        "",
        "## Prior expectations before running",
        "",
        "- RNN-derived reliability plus RNN-only relation encoding should remove the previous shortcut in which the active-rank module can solve the task from exact hand-fed distances after RNN removal.",
        "- Removing RNN should strongly reduce paper-task accuracy if the bottleneck is effective.",
        "- Removing the subject-level commitment/readout should reduce transitive self-consistency and increase circular triads.",
        "- Raw bar observations should be harder than signed-distance observations and may require longer training before matching behavioral findings.",
        "",
        "## Observed checks",
        "",
        *passfail(rows),
        "",
        "## Variant table",
        "",
        "| variant | status | overall | learned | nonlearned | c80 | self-consistency | tau | circular | edge-recon |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def val(k: str) -> str:
            try:
                return f"{float(r.get(k, '')):.3f}"
            except Exception:
                return str(r.get(k, ""))
        lines.append("| " + " | ".join([
            r["variant"], r["status"], val("overall_accuracy"), val("learned_pairs_accuracy"),
            val("nonlearned_pairs_accuracy"), val("consistent_error_subjects_80pct_ratio"),
            val("mean_self_consistency_from_majority_choices"), val("mean_inter_subject_kendall_tau"),
            val("mean_circular_triads_from_majority_choices"), val("edge_recon_loss"),
        ]) + " |")
    (root / "ablation_expectation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {root / 'ablation_summary.csv'} and {root / 'ablation_expectation_report.md'}")


if __name__ == "__main__":
    main()
