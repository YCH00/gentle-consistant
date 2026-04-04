import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Dirichlet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate GENTLE-style interpolation in a 2D toy setting."
    )
    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        metavar=("X", "Y"),
        help="Start point on the 2D line.",
    )
    parser.add_argument(
        "--end",
        nargs=2,
        type=float,
        default=[4.0, 4.0],
        metavar=("X", "Y"),
        help="End point on the 2D line.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Same beta used in GENTLE: alpha = beta * Dirichlet - (beta - 1) / M.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="Number of random GENTLE interpolation samples.",
    )
    parser.add_argument(
        "--num-line-points",
        type=int,
        default=101,
        help="Number of evenly spaced points used to draw the target segment.",
    )
    parser.add_argument(
        "--coverage-bins",
        type=int,
        default=50,
        help="Number of bins used to measure how well samples cover the line segment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for NumPy and PyTorch.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/gentle_2d_interpolation_test.png",
        help="Path to the output figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom plot title.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_gentle_alphas(num_samples: int, beta: float) -> np.ndarray:
    dist = Dirichlet(torch.ones(2))
    alphas = dist.sample((num_samples,))
    alphas = alphas * beta - (beta - 1.0) / 2.0
    return alphas.cpu().numpy()


def build_segment_points(start: np.ndarray, end: np.ndarray, num_points: int) -> np.ndarray:
    lambdas = np.linspace(0.0, 1.0, num_points, dtype=np.float64)
    return (1.0 - lambdas)[:, None] * start[None, :] + lambdas[:, None] * end[None, :]


def project_points_to_line(points: np.ndarray, start: np.ndarray, end: np.ndarray):
    direction = end - start
    direction_norm_sq = float(np.dot(direction, direction))
    if direction_norm_sq <= 0.0:
        raise ValueError("Start and end points must be different.")

    rel = points - start[None, :]
    t_values = rel @ direction / direction_norm_sq
    closest = start[None, :] + t_values[:, None] * direction[None, :]
    distances = np.linalg.norm(points - closest, axis=1)
    return t_values, distances


def compute_metrics(
    gentle_points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    coverage_bins: int,
):
    t_values, distances = project_points_to_line(gentle_points, start, end)
    in_segment_mask = (t_values >= 0.0) & (t_values <= 1.0)

    if np.any(in_segment_mask):
        clipped_t = t_values[in_segment_mask]
        bin_ids = np.floor(clipped_t * coverage_bins).astype(np.int64)
        bin_ids = np.clip(bin_ids, 0, coverage_bins - 1)
        segment_coverage = np.unique(bin_ids).size / coverage_bins
    else:
        segment_coverage = 0.0

    metrics = {
        "max_distance_to_line": float(distances.max()),
        "mean_distance_to_line": float(distances.mean()),
        "t_min": float(t_values.min()),
        "t_max": float(t_values.max()),
        "outside_segment_count": int((~in_segment_mask).sum()),
        "inside_segment_count": int(in_segment_mask.sum()),
        "segment_coverage": float(segment_coverage),
    }
    return metrics, t_values


def plot_results(
    start: np.ndarray,
    end: np.ndarray,
    segment_points: np.ndarray,
    gentle_points: np.ndarray,
    t_values: np.ndarray,
    beta: float,
    metrics: dict,
    output_path: Path,
    title: str,
):
    direction = end - start
    line_start = start - 0.35 * direction
    line_end = end + 0.35 * direction

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        linestyle="--",
        linewidth=1.2,
        color="gray",
        alpha=0.8,
        label="Infinite line",
    )
    ax.plot(
        segment_points[:, 0],
        segment_points[:, 1],
        linewidth=2.0,
        color="tab:blue",
        label="Target segment",
    )

    scatter = ax.scatter(
        gentle_points[:, 0],
        gentle_points[:, 1],
        c=t_values,
        cmap="viridis",
        s=18,
        alpha=0.75,
        label="GENTLE interpolation samples",
    )
    ax.scatter(
        [start[0], end[0]],
        [start[1], end[1]],
        color="tab:red",
        s=90,
        zorder=5,
        label="Endpoints",
    )

    text_lines = [
        f"beta = {beta:.3f}",
        f"max dist = {metrics['max_distance_to_line']:.3e}",
        f"mean dist = {metrics['mean_distance_to_line']:.3e}",
        f"t range = [{metrics['t_min']:.3f}, {metrics['t_max']:.3f}]",
        f"outside segment = {metrics['outside_segment_count']}",
        f"segment coverage = {metrics['segment_coverage']:.2%}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Projection coefficient t")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    start = np.asarray(args.start, dtype=np.float64)
    end = np.asarray(args.end, dtype=np.float64)

    segment_points = build_segment_points(start, end, args.num_line_points)
    alphas = sample_gentle_alphas(args.num_samples, args.beta)
    endpoints = np.stack([start, end], axis=0)
    gentle_points = alphas @ endpoints

    metrics, t_values = compute_metrics(
        gentle_points=gentle_points,
        start=start,
        end=end,
        coverage_bins=args.coverage_bins,
    )

    title = args.title
    if title is None:
        title = "GENTLE 2D interpolation test"

    output_path = Path(args.output)
    plot_results(
        start=start,
        end=end,
        segment_points=segment_points,
        gentle_points=gentle_points,
        t_values=t_values,
        beta=args.beta,
        metrics=metrics,
        output_path=output_path,
        title=title,
    )

    print(f"Start point: {start.tolist()}")
    print(f"End point: {end.tolist()}")
    print(f"Beta: {args.beta}")
    print(f"Num samples: {args.num_samples}")
    print(f"Max distance to line: {metrics['max_distance_to_line']:.6e}")
    print(f"Mean distance to line: {metrics['mean_distance_to_line']:.6e}")
    print(f"Projection range t: [{metrics['t_min']:.6f}, {metrics['t_max']:.6f}]")
    print(f"Samples outside segment: {metrics['outside_segment_count']}")
    print(f"Segment coverage: {metrics['segment_coverage']:.2%}")
    if args.beta == 1.0:
        print("Interpretation: beta=1.0 gives convex interpolation, so samples stay on the line segment between endpoints.")
    else:
        print("Interpretation: beta>1.0 can extrapolate beyond the endpoints while still staying on the same 2D line.")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
