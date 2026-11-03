"""Seed-level aggregation; incomplete cells remain incomplete, never nanmean."""
import csv
import json
import math
from pathlib import Path
import statistics

from . import ARMS, PAIRS
from .common import now, read_json, status, write_json


def complete_statistics(scores, statuses):
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("Nonfinite seed score; do not silently omit failed seeds")
    seeds = [0, 1, 2, 3]
    complete = all(str(seed) in scores and statuses.get(str(seed)) == "completed" for seed in seeds)
    return {"n_seeds": len(scores), "seed_scores": scores, "seed_status": statuses, "complete": complete,
            "mean": statistics.mean(scores[str(seed)] for seed in seeds) if complete else None,
            "sample_std": statistics.stdev(scores[str(seed)] for seed in seeds) if complete else None, "ddof": 1}


def mean(values):
    return statistics.mean(list(values))


def aggregate_main(records, protocol, split="unseen"):
    tasks = {}
    for row in records:
        if row["protocol"] == protocol and row["split"] == split:
            tasks.setdefault(row["task_id"], {}).setdefault(row["repeat"], []).append(row["return"])
    expected_tasks = set(range(10)) if split == "seen" else set(range(10, 20))
    if set(tasks) != expected_tasks:
        raise ValueError("Main-evaluation task coverage incomplete")
    return mean(mean(mean(episodes) for episodes in repeats.values()) for repeats in tasks.values())


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def summarize(args):
    from .artifacts import verify_plan, verify_completion
    root = Path(args.root)
    verify_plan(root)
    protocol, matrix = read_json(root / "protocol.json"), read_json(root / "run_matrix.json")["slots"]
    output = root / "summaries"
    output.mkdir(exist_ok=True)
    arms = [record["arm"] for record in protocol["arms"]]
    for slot in matrix:
        run = root / slot["run_directory"]
        for phase in ("train", "main", "fixed", "probe"):
            marker = run / ("completion.json" if phase == "train" else phase + "_completion.json")
            if marker.exists():
                verify_completion(root, slot, phase)
    cells, pairs, usage, fixed_cells, geometry_cells, probe_cells = [], [], [], [], [], []
    scores_by_key = {}
    for env in protocol["environments"]:
        for arm in arms:
            selected = [row for row in matrix if row["environment"] == env and row["arm"] == arm]
            for main_protocol in ("main_given_context", "main_self_collected"):
                scores, states = {}, {}
                for slot in selected:
                    run, seed = root / slot["run_directory"], str(slot["seed"])
                    result = run / "evaluation/main_itr_499.json"
                    states[seed] = slot["status"]
                    if (run / "completion.json").exists() and result.exists():
                        data = read_json(result)
                        if data["kind"] != "formal_main":
                            raise ValueError("Smoke data encountered in formal results")
                        scores[seed] = aggregate_main(data["records"], main_protocol)
                        states[seed] = "completed"
                cell = {"environment": env, "arm": arm, "protocol": main_protocol, **complete_statistics(scores, states)}
                cells.append(cell)
                scores_by_key[env, arm, main_protocol] = cell
            for slot in selected:
                run = root / slot["run_directory"]
                if (run / "completion.json").exists():
                    data = read_json(run / "completion.json")
                    usage.append({"environment": env, "arm": arm, "seed": slot["seed"],
                                  "main_updates": data["main_updates"], "update_counts": data["update_counts"],
                                  "seconds_last_process": data["wall_seconds_this_process"], "peak_cuda_bytes": data["peak_cuda_bytes"],
                                  "runtime_totals_committed_iterations": data["runtime_totals_committed_iterations"],
                                  "virtual_usage": data["virtual_usage"],
                                  "initial_sha256": data["initial_sha256"]})
            for split in ("seen", "unseen"):
                for condition in ("offline", "policy499_sigma000", "policy499_sigma020"):
                    scores, states = {}, {}
                    for slot in selected:
                        run, seed = root / slot["run_directory"], str(slot["seed"])
                        path = run / "evaluation/fixed_context.json"
                        states[seed] = "missing_fixed_context"
                        if path.exists() and (run / "fixed_completion.json").exists():
                            data = read_json(path)
                            if data["kind"] != "formal_fixed":
                                raise ValueError("Non-formal fixed-context records")
                            tasks = {}
                            for row in data["records"]:
                                if row["split"] == split and row["condition"] == condition:
                                    tasks.setdefault(row["task_id"], []).append(row["return"])
                            if len(tasks) != 10 or any(len(values) != 60 for values in tasks.values()):
                                raise ValueError("Fixed-context coverage incomplete")
                            scores[seed] = mean(mean(values) for values in tasks.values())
                            states[seed] = "completed"
                    fixed_cells.append({"environment": env, "arm": arm, "split": split, "condition": condition, **complete_statistics(scores, states)})
                for metric in ("B", "S", "W", "B_over_S", "identification_accuracy"):
                    for condition in ("offline", "policy499_sigma000", "policy499_sigma020"):
                        scores, states = {}, {}
                        for slot in selected:
                            run, seed = root / slot["run_directory"], str(slot["seed"])
                            path = run / "evaluation/representation_metrics.json"
                            states[seed] = "missing_representation"
                            if path.exists() and (run / "fixed_completion.json").exists():
                                row = next(item for item in read_json(path)[split] if item["reference"] == "policy499_sigma000" and item["condition"] == condition)
                                if row[metric] is not None:
                                    scores[seed], states[seed] = row[metric], "completed"
                                else:
                                    states[seed] = "undefined_collapse"
                        geometry_cells.append({"environment": env, "arm": arm, "split": split, "condition": condition,
                                               "metric": metric, **complete_statistics(scores, states)})
                for metric in ("reward_MSE", "reward_NMSE", "dynamics_MSE", "dynamics_NMSE"):
                    scores, states = {}, {}
                    for slot in selected:
                        run, seed = root / slot["run_directory"], str(slot["seed"])
                        path = run / "probes/results.json"
                        states[seed] = "missing_probe"
                        if path.exists() and (run / "probe_completion.json").exists():
                            data = read_json(path)
                            values = [row[metric] for row in data["metrics"] if row["split"] == split]
                            if len(values) != 10:
                                raise ValueError("Probe task coverage incomplete")
                            scores[seed], states[seed] = mean(values), "completed"
                    probe_cells.append({"environment": env, "arm": arm, "split": split, "metric": metric, **complete_statistics(scores, states)})
        for left, right in PAIRS:
            for main_protocol in ("main_given_context", "main_self_collected"):
                a, b = scores_by_key[env, left, main_protocol], scores_by_key[env, right, main_protocol]
                common = set(a["seed_scores"]) & set(b["seed_scores"])
                scores = {seed: a["seed_scores"][seed] - b["seed_scores"][seed] for seed in common}
                states = {str(seed): "completed" if str(seed) in scores else "missing_paired_seed" for seed in (0, 1, 2, 3)}
                pairs.append({"environment": env, "comparison": left + " - " + right, "protocol": main_protocol, **complete_statistics(scores, states)})
    fixed_lookup = {(row["environment"], row["arm"], row["split"], row["condition"]): row for row in fixed_cells}
    noise_differences, fixed_pairs = [], []
    def paired_row(left, right):
        available = set(left["seed_scores"]) & set(right["seed_scores"])
        differences = {seed: left["seed_scores"][seed] - right["seed_scores"][seed] for seed in available}
        return complete_statistics(differences, {str(seed): "completed" if str(seed) in available else "missing_paired_seed" for seed in range(4)})
    for env in protocol["environments"]:
        for split in ("seen", "unseen"):
            for arm in arms:
                noise_differences.append({"environment": env, "arm": arm, "split": split,
                                          "comparison": "policy499_sigma020 - policy499_sigma000",
                                          **paired_row(fixed_lookup[env, arm, split, "policy499_sigma020"], fixed_lookup[env, arm, split, "policy499_sigma000"])})
            for left, right in PAIRS:
                for condition in ("offline", "policy499_sigma000", "policy499_sigma020"):
                    fixed_pairs.append({"environment": env, "split": split, "condition": condition,
                                        "comparison": left + " - " + right,
                                        **paired_row(fixed_lookup[env, left, split, condition], fixed_lookup[env, right, split, condition])})
    for name, rows in (("main_results", cells), ("paired_differences", pairs), ("P_usage_cost", usage),
                       ("R_noise_return_change", noise_differences), ("fixed_context_paired_differences", fixed_pairs),
                       ("fixed_context_returns", fixed_cells), ("R_representation", geometry_cells), ("V_feedback_prediction", probe_cells)):
        write_json(output / (name + ".json"), rows)
        write_csv(output / (name + ".csv"), rows)
    def formatted(cell):
        return "incomplete (%d/4)" % cell["n_seeds"] if not cell["complete"] else "%.3f ± %.3f" % (cell["mean"], cell["sample_std"])
    headings = ["arm", "Cheetah Given-context", "Cheetah One-shot", "Walker Given-context", "Walker One-shot"]
    lines = ["# Component ablations — matched loss family", "", "P includes virtual critic, actor-Q and actor-BC. Four training seeds; sample std (ddof=1).", "", "| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * 5) + " |"]
    latex = ["\\begin{tabular}{lrrrr}", " & ".join(headings) + " \\\\"]
    for arm in arms:
        values = [formatted(scores_by_key[env, arm, main_protocol]) for env in protocol["environments"] for main_protocol in ("main_given_context", "main_self_collected")]
        label = "Matched backbone" if arm == "backbone" else "Matched full" if arm == "full" else arm
        lines.append("| " + " | ".join([label] + values) + " |")
        latex.append(" & ".join([label.replace("_", "\\_")] + values) + " \\\\")
    latex.append("\\end{tabular}")
    (output / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "main_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    write_json(output / "completion.json", {"created_at": now(), "core_main_complete_cells": sum(c["complete"] for c in cells if c["arm"] in ARMS),
                                             "expected_core_main_cells": 24, "formal_training_complete": sum(row["status"] == "completed" for row in matrix),
                                             "expected_core_training_slots": 48, "diagnostic_and_probe_complete": all(c["complete"] for c in fixed_cells + probe_cells)})
    status(root, summaries="Generated from actual records only; missing cells explicitly incomplete", D={"core_main_complete_cells": sum(c["complete"] for c in cells if c["arm"] in ARMS), "expected": 24,
                                                                                                           "formal_bank_and_probes_pending": not all(c["complete"] for c in fixed_cells + probe_cells)})
    print("Wrote summaries:", output)
