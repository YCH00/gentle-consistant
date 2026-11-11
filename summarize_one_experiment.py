"""Summarize one experiment using CSV, TensorBoard, then formatted debug logs."""

import argparse
import csv
import json
import math
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path


ENVS = ("ant-dir", "point-robot", "cheetah-dir", "cheetah-vel",
        "hopper-rand-params", "walker-rand-params")
METRICS = {
    "given_context": "AverageReturn_all_test_tasks",
    "self_collected": "AverageReturn_all_test_tasks_expl",
}
BEIJING = timezone(timedelta(hours=8))


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite return")
    return value


def read_debug_pair(run, epoch, note):
    path = run / "debug.log"
    if not path.exists():
        raise ValueError(note + "; debug.log missing")
    pending = {mode: [] for mode in METRICS}
    blocks = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            for mode, column in METRICS.items():
                match = re.match(r"^\s*\|?\s*" + re.escape(column) +
                                 r"\s+\|?\s*([^\s|]+)\s*\|?\s*$", line)
                if match:
                    pending[mode].append((match.group(1), number))
            match = re.search(r"\bEpoch\s*\|?\s*(\d+)(?:\.0+)?\s*(?:\|\s*)?$", line)
            if match:
                if int(match.group(1)) == epoch:
                    blocks.append((number, pending))
                pending = {mode: [] for mode in METRICS}
    if len(blocks) != 1:
        raise ValueError(note + "; expected one debug.log block at Epoch=%d, found %d" %
                         (epoch, len(blocks)))
    epoch_line, points = blocks[0]
    result = {}
    for mode in METRICS:
        if len(points[mode]) != 1:
            raise ValueError("debug.log Epoch=%d block has %d %s records" %
                             (epoch, len(points[mode]), mode))
        value, number = points[mode][0]
        result[mode] = dict(
            return_value=finite(value), source_type="debug_log", source_file=str(path),
            event_time="", note=note + "; metric line=%d; Epoch line=%d; " %
            (number, epoch_line) +
            "formatted debug.log value; original full precision unavailable")
    return result


def read_pair(run, epoch):
    path = run / "progress.csv"
    note = "progress.csv missing"
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, strict=True)
            if "Epoch" not in (reader.fieldnames or []):
                raise ValueError("progress.csv has no Epoch column")
            rows = list(reader)
        epochs = [finite(row["Epoch"]) for row in rows]
        matches = [row for row, step in zip(rows, epochs) if step == epoch]
        if len(matches) > 1:
            raise ValueError("progress.csv has duplicate Epoch=%d rows" % epoch)
        if matches:
            return {
                mode: dict(return_value=finite(matches[0][column]),
                           source_type="csv", source_file=str(path),
                           event_time="", note="")
                for mode, column in METRICS.items()
            }
        note = "CSV has no Epoch=%d; maximum recorded epoch=%s" % (
            epoch, max(epochs) if epochs else "none")

    event_files = sorted(run.glob("events.out.tfevents.*"))
    if not event_files:
        return read_debug_pair(run, epoch, note + "; no TensorBoard event files")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    candidates = []
    for path in event_files:
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0},
                               purge_orphaned_data=False)
        acc.Reload()
        available = acc.Tags().get("scalars", [])
        points = {}
        for mode, column in METRICS.items():
            tag = "Return/" + column
            points[mode] = [e for e in acc.Scalars(tag) if e.step == epoch] \
                if tag in available else []
        if any(points.values()):
            candidates.append((path, points))

    if not candidates:
        return read_debug_pair(run, epoch, note + "; no TensorBoard step=%d records" % epoch)
    if len(candidates) != 1:
        names = ", ".join(path.name for path, _ in candidates) or "none"
        raise ValueError(note + "; expected one TensorBoard file containing "
                         "step=%d, found %d (%s)" % (epoch, len(candidates), names))
    path, points = candidates[0]
    result = {}
    for mode in METRICS:
        if len(points[mode]) != 1:
            raise ValueError("%s: %s has %d records at step=%d" %
                             (path.name, mode, len(points[mode]), epoch))
        event = points[mode][0]
        result[mode] = dict(
            return_value=finite(event.value), source_type="tensorboard",
            source_file=str(path),
            event_time=datetime.fromtimestamp(event.wall_time, BEIJING).isoformat(),
            note=note)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="For example: 23_local_decay")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--epoch", type=int, default=499)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--log-group", default="gentle",
                        help="Log folder below each environment; mixed runs use mixed_rme_v3_gentle")
    parser.add_argument("--envs", nargs="+", choices=ENVS, default=list(ENVS))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    if Path(args.experiment).name != args.experiment or args.experiment in (".", ".."):
        parser.error("experiment must be a single folder name")
    if Path(args.log_group).name != args.log_group or args.log_group in (".", ".."):
        parser.error("--log-group must be a single folder name")
    if len(set(args.envs)) != len(args.envs):
        parser.error("--envs must not contain duplicates")
    root = args.root.resolve()
    out = root / "result" / (args.experiment + "_summary")
    details, summary = [], []
    print("Experiment: %s | Epoch: %d" % (args.experiment, args.epoch))

    for env in args.envs:
        for seed in args.seeds:
            run = root / "logs" / env / args.log_group / ("seed%d" % seed) / args.experiment
            error, pair = "", {}
            try:
                pair = read_pair(run, args.epoch)
                source = pair["given_context"]
                if source["source_type"] != "csv":
                    print("RECOVERED (%s): %s seed%d <- %s" %
                          (source["source_type"], env, seed, source["source_file"]))
                    if source["source_type"] == "debug_log":
                        print("NOTE: debug.log stores formatted, rounded values.")
            except Exception as exc:
                error = "%s: %s" % (type(exc).__name__, exc)
                print("ERROR: %s seed%d: %s" % (env, seed, error))
            for mode in METRICS:
                value = pair.get(mode, {})
                details.append(dict(
                    experiment=args.experiment, environment=env, evaluation=mode,
                    seed=seed, epoch=args.epoch,
                    return_value=value.get("return_value"),
                    status="error" if error else "ok", error=error,
                    source_type=value.get("source_type", ""),
                    source_file=value.get("source_file", ""),
                    event_time=value.get("event_time", ""),
                    note=value.get("note", "")))

    print("\nEnvironment               Evaluation       n       Mean +/- sample SD")
    for env in args.envs:
        for mode in METRICS:
            values = [r["return_value"] for r in details if r["environment"] == env
                      and r["evaluation"] == mode and r["status"] == "ok"]
            n = len(values)
            mean = statistics.mean(values) if n else None
            sd = statistics.stdev(values) if n > 1 else None
            summary.append(dict(
                experiment=args.experiment, environment=env, evaluation=mode,
                epoch=args.epoch, n=n, expected_n=len(args.seeds),
                complete=n == len(args.seeds), mean_return=mean, sample_std=sd))
            text = "NA" if mean is None else "%.2f +/- %s" % (
                mean, "NA" if sd is None else "%.2f" % sd)
            print("%-25s %-16s %d/%d     %s%s" %
                  (env, mode, n, len(args.seeds), text,
                   "" if n == len(args.seeds) else " [INCOMPLETE]"))

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("summary.csv", summary), ("per_seed.csv", details)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(dict(experiment=args.experiment, root=str(root), log_group=args.log_group,
                       environments=args.envs, epoch=args.epoch,
                       seeds=args.seeds, summary=summary, per_seed=details),
                  f, indent=2, allow_nan=False)
    print("Saved: " + str(out))


if __name__ == "__main__":
    main()
