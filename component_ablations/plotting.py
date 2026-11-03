"""Publication exports made only from complete, verified seed results."""
from pathlib import Path
import statistics
from . import ARMS
from .common import now, read_json, sha256, status, write_json
from .reporting import aggregate_main, summarize


def plot(args):
    summarize(args)
    root = Path(args.root)
    cells = read_json(root / "summaries/main_results.json")
    if not any(row["complete"] for row in cells):
        status(root, E="PENDING_RESULTS: no complete four-seed cells; no performance figures generated")
        print("No complete four-seed results; no performance figures generated.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    protocol = read_json(root / "protocol.json")
    matrix = read_json(root / "run_matrix.json")["slots"]
    output = root / "figures"
    output.mkdir(exist_ok=True)
    inputs, exports = [], []
    names = {"backbone": "Backbone", "real_only": "R only", "no_real": "w/o R",
             "no_virtual_recovery": "w/o V", "no_virtual_rl": "w/o P", "full": "Matched full",
             "full_global": "Global", "full_constant": "Constant weight"}

    def save(figure, stem):
        figure.tight_layout()
        for suffix in ("pdf", "svg", "png"):
            path = output / (stem + "." + suffix)
            figure.savefig(path, dpi=300, bbox_inches="tight")
            exports.append({"path": str(path.relative_to(root)), "sha256": sha256(path)})
        plt.close(figure)

    for env in protocol["environments"]:
        for evaluation in ("main_given_context", "main_self_collected"):
            selected = [row for row in cells if row["environment"] == env and row["protocol"] == evaluation and row["complete"]]
            if not set(ARMS) <= {row["arm"] for row in selected}:
                continue
            figure, axis = plt.subplots(figsize=(10, 4))
            axis.bar(range(len(selected)), [row["mean"] for row in selected], yerr=[row["sample_std"] for row in selected], capsize=4)
            axis.set_xticks(range(len(selected)))
            axis.set_xticklabels([names[row["arm"]] for row in selected], rotation=20, ha="right")
            axis.set_ylabel("Return (mean ± sample std; 4 training seeds)")
            axis.set_title(env + " — " + evaluation.replace("main_", "") + " — iteration 499")
            inputs.append({"figure": env + "_" + evaluation, "cells": selected})
            save(figure, env + "_" + evaluation)
            figure, axis = plt.subplots(figsize=(8, 4))
            curve_inputs = []
            for record in protocol["arms"]:
                arm = record["arm"]
                curves = []
                for iteration in protocol["checkpoints"]:
                    scores = []
                    for seed in range(4):
                        slot = next(row for row in matrix if row["environment"] == env and row["arm"] == arm and row["seed"] == seed)
                        path = root / slot["run_directory"] / ("evaluation/main_itr_%d.json" % iteration)
                        if not path.exists():
                            break
                        data = read_json(path)
                        if data["kind"] != "formal_main":
                            raise ValueError("Non-formal curve data")
                        scores.append(aggregate_main(data["records"], evaluation))
                    if len(scores) != 4:
                        curves = []
                        break
                    curves.append({"iteration": iteration, "seed_scores": scores, "mean": statistics.mean(scores), "sample_std": statistics.stdev(scores)})
                if curves:
                    axis.errorbar([row["iteration"] for row in curves], [row["mean"] for row in curves],
                                  yerr=[row["sample_std"] for row in curves], label=names[arm], marker="o", capsize=3)
                    curve_inputs.append({"arm": arm, "checkpoints": curves})
            axis.set_xlabel("Training iteration")
            axis.set_ylabel("Return (mean ± sample std; 4 training seeds)")
            axis.set_title(env + " — " + evaluation.replace("main_", ""))
            axis.legend(fontsize=8)
            inputs.append({"figure": env + "_" + evaluation + "_curve", "curves": curve_inputs})
            save(figure, env + "_" + evaluation + "_curve")
    write_json(output / "plot_inputs.json", inputs)
    write_json(output / "manifest.json", {"created_at": now(), "exports": exports, "matplotlib": matplotlib.__version__,
                                          "checkpoint_selection": "fixed 499; curves at 0/200/400/499", "error_bars": "training-seed sample std, ddof=1"})
    status(root, E={"figures_exported": len(exports), "source": "figures/plot_inputs.json", "supplemental_complete": all(row["complete"] for row in cells)})
