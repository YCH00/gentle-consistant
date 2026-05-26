import argparse
import csv
import re
from pathlib import Path


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def natural_key(text):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(text))]


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def has_event_file(path):
    return any(path.rglob("events.out.tfevents.*"))


def newest_mtime(path):
    event_files = list(path.rglob("events.out.tfevents.*"))
    if not event_files:
        return path.stat().st_mtime
    return max(event_file.stat().st_mtime for event_file in event_files)


def select_one_run(seed_dir, experiment_pattern, pick):
    matches = [
        path
        for path in seed_dir.glob(experiment_pattern)
        if path.is_dir() and has_event_file(path)
    ]
    matches = sorted(matches, key=lambda path: natural_key(path.name))

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if pick == "latest":
        return max(matches, key=newest_mtime)
    if pick == "first":
        return matches[0]

    match_text = "\n".join(f"  - {path}" for path in matches)
    raise ValueError(
        f"Pattern '{experiment_pattern}' matched multiple runs under {seed_dir}:\n"
        f"{match_text}\nUse a more specific pattern or pass --pick latest/first."
    )


def find_seed_runs(root, experiment_pattern, seed_pattern, pick):
    root = Path(root)
    seed_dirs = sorted(
        [path for path in root.glob(seed_pattern) if path.is_dir()],
        key=lambda path: natural_key(path.name),
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories matched '{seed_pattern}' under {root}")

    runs = []
    missing = []
    for seed_dir in seed_dirs:
        run_dir = select_one_run(seed_dir, experiment_pattern, pick)
        if run_dir is None:
            missing.append(seed_dir.name)
        else:
            runs.append((seed_dir.name, run_dir))

    if not runs:
        raise FileNotFoundError(
            f"No TensorBoard runs matched experiment pattern '{experiment_pattern}' under {root}"
        )
    if missing:
        print("Warning: these seed directories had no matching event files:")
        for seed_name in missing:
            print(f"  - {seed_name}")
    return runs


def load_event_accumulator(run_dir):
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(
        str(run_dir), size_guidance={event_accumulator.SCALARS: 0}
    )
    accumulator.Reload()
    return accumulator


def list_scalar_tags(runs):
    run_tags = {}
    for seed_name, run_dir in runs:
        accumulator = load_event_accumulator(run_dir)
        tags = set(accumulator.Tags().get("scalars", []))
        run_tags[(seed_name, run_dir)] = tags

    common_tags = sorted(set.intersection(*run_tags.values())) if run_tags else []
    all_tags = sorted(set.union(*run_tags.values())) if run_tags else []

    print("Matched runs:")
    for seed_name, run_dir in runs:
        print(f"  {seed_name}: {run_dir}")
    print("\nScalar tags common to every matched seed:")
    for tag in common_tags:
        print(f"  {tag}")

    uncommon_tags = [tag for tag in all_tags if tag not in common_tags]
    if uncommon_tags:
        print("\nScalar tags not present in every seed:")
        for tag in uncommon_tags:
            print(f"  {tag}")


def load_scalar_series(seed_name, run_dir, tag):
    accumulator = load_event_accumulator(run_dir)
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        raise KeyError(
            f"Tag '{tag}' was not found in {run_dir}. "
            "Run again with --list-tags to see available scalar tags."
        )

    values_by_step = {}
    for event in accumulator.Scalars(tag):
        values_by_step[int(event.step)] = float(event.value)

    if not values_by_step:
        raise ValueError(f"Tag '{tag}' has no scalar values in {run_dir}")

    steps = np.array(sorted(values_by_step), dtype=float)
    values = np.array([values_by_step[int(step)] for step in steps], dtype=float)
    return {"seed": seed_name, "run_dir": run_dir, "steps": steps, "values": values}


def moving_average(values, window):
    if window <= 1:
        return values
    if window > len(values):
        raise ValueError(
            f"--smooth-window={window} is larger than the number of plotted points ({len(values)})"
        )
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def align_series(series, align):
    step_sets = [set(item["steps"].astype(int)) for item in series]
    if align == "intersection":
        aligned_steps = sorted(set.intersection(*step_sets))
        if not aligned_steps:
            raise ValueError(
                "No common steps across seeds. Use --align union or --align interpolate."
            )
        steps = np.array(aligned_steps, dtype=float)
        matrix = np.vstack(
            [
                np.array(
                    [
                        dict(zip(item["steps"].astype(int), item["values"]))[int(step)]
                        for step in steps
                    ],
                    dtype=float,
                )
                for item in series
            ]
        )
        return steps, matrix

    if align == "union":
        steps = np.array(sorted(set.union(*step_sets)), dtype=float)
        matrix = np.full((len(series), len(steps)), np.nan, dtype=float)
        step_to_col = {int(step): idx for idx, step in enumerate(steps)}
        for row_idx, item in enumerate(series):
            for step, value in zip(item["steps"].astype(int), item["values"]):
                matrix[row_idx, step_to_col[int(step)]] = value
        return steps, matrix

    min_step = max(item["steps"][0] for item in series)
    max_step = min(item["steps"][-1] for item in series)
    steps = np.array(
        sorted(step for step in set.union(*step_sets) if min_step <= step <= max_step),
        dtype=float,
    )
    if len(steps) == 0:
        raise ValueError("No overlapping step range across seeds for interpolation.")
    matrix = np.vstack(
        [np.interp(steps, item["steps"], item["values"]) for item in series]
    )
    return steps, matrix


def summarize(series, align):
    steps, matrix = align_series(series, align)
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    counts = np.sum(~np.isnan(matrix), axis=0)
    return steps, matrix, mean, std, counts


def save_csv(csv_path, steps, mean, std, counts):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["step", "mean", "std", "num_seeds"])
        for row in zip(steps.astype(int), mean, std, counts.astype(int)):
            writer.writerow(row)


def save_compare_csv(csv_path, summaries):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["experiment", "label", "step", "mean", "std", "num_seeds"])
        for summary in summaries:
            for row in zip(
                summary["steps"].astype(int),
                summary["mean"],
                summary["std"],
                summary["counts"].astype(int),
            ):
                writer.writerow([summary["experiment"], summary["label"], *row])


def default_output_path(experiment, tag):
    output_name = f"{sanitize_filename(experiment)}_{sanitize_filename(tag)}_mean.png"
    return Path("figures") / output_name


def default_compare_output_path(experiments, tag):
    experiment_text = "_vs_".join(sanitize_filename(experiment) for experiment in experiments)
    output_name = f"{experiment_text}_{sanitize_filename(tag)}_compare.png"
    return Path("figures") / output_name


def plot_average(
    output_path,
    series,
    steps,
    matrix,
    mean,
    std,
    tag,
    title,
    ylabel,
    show_seeds,
    smooth_window,
):
    plt = get_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_mean = moving_average(mean, smooth_window)
    plot_std = moving_average(std, smooth_window)

    fig, ax = plt.subplots(figsize=(8, 5))

    if show_seeds:
        for item, values in zip(series, matrix):
            ax.plot(
                steps,
                values,
                color="#9ca3af",
                linewidth=1.0,
                alpha=0.35,
                label=item["seed"],
            )

    ax.plot(steps, plot_mean, color="#2563eb", linewidth=2.3, label="mean")
    ax.fill_between(
        steps,
        plot_mean - plot_std,
        plot_mean + plot_std,
        color="#2563eb",
        alpha=0.18,
        label="std",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or tag.split("/")[-1])
    ax.set_title(title or tag)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_compare(output_path, summaries, tag, title, ylabel, smooth_window, no_std_shade):
    plt = get_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for summary in summaries:
        steps = summary["steps"]
        plot_mean = moving_average(summary["mean"], smooth_window)
        plot_std = moving_average(summary["std"], smooth_window)
        line = ax.plot(
            steps,
            plot_mean,
            linewidth=2.3,
            label=summary["label"],
        )[0]
        if not no_std_shade:
            ax.fill_between(
                steps,
                plot_mean - plot_std,
                plot_mean + plot_std,
                color=line.get_color(),
                alpha=0.14,
            )

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel or tag.split("/")[-1])
    ax.set_title(title or tag)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average TensorBoard scalar curves across seed directories."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Directory containing seed*/ subdirectories, e.g. logs/ant-dir/gentle.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        action="append",
        help=(
            "Run directory name or glob under each seed directory. "
            "Use quotes for wildcards, e.g. '*vt_policy_loss'. "
            "Pass this option multiple times to compare experiments."
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Legend label for one experiment. Repeat once per --experiment.",
    )
    parser.add_argument(
        "--tag",
        default="Return/AverageReturn_all_test_tasks",
        help="TensorBoard scalar tag to average.",
    )
    parser.add_argument(
        "--seed-pattern",
        default="seed*",
        help="Glob for seed directories under --root. Default: seed*.",
    )
    parser.add_argument(
        "--pick",
        choices=["error", "latest", "first"],
        default="error",
        help="What to do if --experiment matches multiple runs in one seed directory.",
    )
    parser.add_argument(
        "--align",
        choices=["intersection", "union", "interpolate"],
        default="intersection",
        help="How to align scalar steps before averaging.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. Default: figures/<experiment>_<tag>_mean.png.",
    )
    parser.add_argument(
        "--csv-output",
        default=None,
        help="Optional output CSV path for step/mean/std/num_seeds.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered moving-average window applied to the plotted mean/std.",
    )
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="Also draw individual seed curves in light gray for single-experiment plots.",
    )
    parser.add_argument(
        "--no-std-shade",
        action="store_true",
        help="Do not draw standard-deviation shading in multi-experiment compare plots.",
    )
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--ylabel", default=None, help="Optional y-axis label.")
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List scalar tags in the matched runs and exit.",
    )
    return parser.parse_args()


def labels_for_experiments(experiments, labels):
    if labels is None:
        return experiments
    if len(labels) != len(experiments):
        raise ValueError("--label must be provided once for each --experiment.")
    return labels


def summarize_experiment(args, experiment, label, runs=None):
    if runs is None:
        runs = find_seed_runs(args.root, experiment, args.seed_pattern, args.pick)
    series = [load_scalar_series(seed_name, run_dir, args.tag) for seed_name, run_dir in runs]
    steps, matrix, mean, std, counts = summarize(series, args.align)
    return {
        "experiment": experiment,
        "label": label,
        "runs": runs,
        "series": series,
        "steps": steps,
        "matrix": matrix,
        "mean": mean,
        "std": std,
        "counts": counts,
    }


def run_single_experiment(args):
    experiment = args.experiment[0]
    runs = find_seed_runs(args.root, experiment, args.seed_pattern, args.pick)

    if args.list_tags:
        list_scalar_tags(runs)
        return

    summary = summarize_experiment(args, experiment, experiment, runs=runs)

    if args.output is None:
        output_path = default_output_path(experiment, args.tag)
    else:
        output_path = Path(args.output)

    plot_average(
        output_path=output_path,
        series=summary["series"],
        steps=summary["steps"],
        matrix=summary["matrix"],
        mean=summary["mean"],
        std=summary["std"],
        tag=args.tag,
        title=args.title,
        ylabel=args.ylabel,
        show_seeds=args.show_seeds,
        smooth_window=args.smooth_window,
    )

    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    save_csv(csv_path, summary["steps"], summary["mean"], summary["std"], summary["counts"])

    print("Matched runs:")
    for seed_name, run_dir in runs:
        print(f"  {seed_name}: {run_dir}")
    print(f"Saved figure: {output_path}")
    print(f"Saved CSV: {csv_path}")


def run_compare_experiments(args):
    experiments = args.experiment
    labels = labels_for_experiments(experiments, args.label)
    experiment_runs = {
        experiment: find_seed_runs(args.root, experiment, args.seed_pattern, args.pick)
        for experiment in experiments
    }

    if args.list_tags:
        for experiment in experiments:
            print(f"\n=== {experiment} ===")
            list_scalar_tags(experiment_runs[experiment])
        return

    summaries = [
        summarize_experiment(args, experiment, label, runs=experiment_runs[experiment])
        for experiment, label in zip(experiments, labels)
    ]

    if args.output is None:
        output_path = default_compare_output_path(experiments, args.tag)
    else:
        output_path = Path(args.output)

    plot_compare(
        output_path=output_path,
        summaries=summaries,
        tag=args.tag,
        title=args.title,
        ylabel=args.ylabel,
        smooth_window=args.smooth_window,
        no_std_shade=args.no_std_shade,
    )

    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    save_compare_csv(csv_path, summaries)

    print("Matched runs:")
    for experiment, label in zip(experiments, labels):
        print(f"  {label} ({experiment}):")
        for seed_name, run_dir in experiment_runs[experiment]:
            print(f"    {seed_name}: {run_dir}")
    print(f"Saved compare figure: {output_path}")
    print(f"Saved compare CSV: {csv_path}")


def main():
    args = parse_args()
    global np
    import numpy as np

    if len(args.experiment) == 1:
        run_single_experiment(args)
    else:
        run_compare_experiments(args)


if __name__ == "__main__":
    main()
