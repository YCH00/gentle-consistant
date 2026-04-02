import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def plot_task_embeddings(
    real_zs,
    output_path,
    title,
    virtual_zs=None,
    real_legend_label='Real tasks',
    virtual_legend_label='Virtual tasks',
    tsne_seed=None,
):
    real_zs = np.asarray(real_zs)
    if real_zs.ndim != 3:
        raise ValueError(f'Expected real_zs to have shape [n_tasks, n_points, z_dim], got {real_zs.shape}')

    n_tasks, n_points, _ = real_zs.shape
    plot_arrays = [real_zs.reshape(n_tasks * n_points, -1)]

    if virtual_zs is not None:
        virtual_zs = np.asarray(virtual_zs)
        if virtual_zs.ndim == 2:
            virtual_zs = virtual_zs[np.newaxis, ...]
        if virtual_zs.ndim != 3:
            raise ValueError(
                f'Expected virtual_zs to have shape [n_groups, n_points, z_dim] or [n_points, z_dim], got {virtual_zs.shape}'
            )
        plot_arrays.append(virtual_zs.reshape(-1, virtual_zs.shape[-1]))

    tsne = TSNE(n_components=2, random_state=tsne_seed)
    proj = tsne.fit_transform(np.concatenate(plot_arrays, axis=0))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    real_proj = proj[:n_tasks * n_points]
    real_colors = plt.cm.tab20(np.linspace(0, 1, max(n_tasks, 1)))
    for task_idx in range(n_tasks):
        idxs = np.arange(task_idx * n_points, (task_idx + 1) * n_points)
        ax.scatter(
            real_proj[idxs, 0],
            real_proj[idxs, 1],
            s=6,
            alpha=0.35,
            color=real_colors[task_idx],
            marker='o',
        )

    if virtual_zs is not None:
        n_virtual_groups, n_virtual_points, _ = virtual_zs.shape
        virtual_proj = proj[n_tasks * n_points:]
        for virtual_idx in range(n_virtual_groups):
            start = virtual_idx * n_virtual_points
            end = (virtual_idx + 1) * n_virtual_points
            ax.scatter(
                virtual_proj[start:end, 0],
                virtual_proj[start:end, 1],
                s=18,
                alpha=0.85,
                color='black',
                marker='x',
                linewidths=0.9,
            )

    ax.scatter([], [], s=18, alpha=0.35, color='gray', marker='o', label=real_legend_label)
    if virtual_zs is not None:
        ax.scatter([], [], s=28, alpha=0.85, color='black', marker='x', linewidths=0.9, label=virtual_legend_label)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel('t-SNE dimension 1', fontsize=15)
    ax.set_ylabel('t-SNE dimension 2', fontsize=15)
    ax.legend(loc='best', fontsize=12)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def build_default_output_path(real_path):
    real_path = Path(real_path)
    return real_path.with_suffix('.png')


def main():
    parser = argparse.ArgumentParser(description='Re-plot saved task embeddings with t-SNE.')
    parser.add_argument('--real', required=True, help='Path to real task embeddings .npy file.')
    parser.add_argument('--virtual', default=None, help='Optional path to virtual task embeddings .npy file.')
    parser.add_argument('--output', default=None, help='Output png path. Defaults to replacing .npy with .png.')
    parser.add_argument('--title', default=None, help='Figure title. Defaults to the real .npy stem.')
    parser.add_argument('--tsne-seed', type=int, default=0, help='Random seed for t-SNE. Use the same seed to reproduce the same layout.')
    args = parser.parse_args()

    real_path = Path(args.real)
    virtual_path = Path(args.virtual) if args.virtual is not None else None
    output_path = Path(args.output) if args.output is not None else build_default_output_path(real_path)
    title = args.title if args.title is not None else real_path.stem

    real_zs = np.load(real_path)
    virtual_zs = np.load(virtual_path) if virtual_path is not None and virtual_path.exists() else None

    plot_task_embeddings(
        real_zs=real_zs,
        virtual_zs=virtual_zs,
        output_path=output_path,
        title=title,
        tsne_seed=args.tsne_seed,
    )

    print(f'Saved t-SNE figure to {output_path}')


if __name__ == '__main__':
    main()
