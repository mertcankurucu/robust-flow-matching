"""Compact paper figure and separate PowerPoint panels, from saved arrays."""

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#ff7f00",
    "#984ea3",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
]


def read_run(out, mode, method):
    matches = []
    for result_path in (out / mode).glob("*/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["config"]["method"] == method:
            with np.load(
                result_path.with_name("figure_data.npz"), allow_pickle=False
            ) as data:
                matches.append((dict(data), result))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {mode}/{method} run, found {len(matches)} in {out}"
        )
    return matches[0]


def subset(points, count, seed, labels=None):
    # Sample across the entire array: training pools append outliers and
    # generated conditional arrays are class-blocked, so prefix slices mislead.
    indices = np.random.default_rng(seed).choice(
        len(points), min(count, len(points)), replace=False
    )
    return points[indices], None if labels is None else labels[indices]


def save_figure(fig, stem, tight=True):
    for ext in ("pdf", "png"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(
            path, dpi=300, bbox_inches="tight" if tight else None, pad_inches=0.025
        )
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure was not written: {path}")
        print(f"Saved: {path}")


def plot_figure(out, n_plot, layout="row", figure_name="spirals_figure_y_v2"):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    xb, xb_result = read_run(out, "2d_xnoise", "ot_fm")
    xr, xr_result = read_run(out, "2d_xnoise", "fm_trs")
    yb, yb_result = read_run(out, "2d_ynoise", "ot_fm")
    yr, yr_result = read_run(out, "2d_ynoise", "fm_ccs")
    # Record whether each compared method pair used identical training pools.
    manifest["compared_training_pools_identical"] = {
        "x": bool(np.array_equal(xb["training"], xr["training"])),
        "y": bool(
            np.array_equal(yb["training"], yr["training"])
            and np.array_equal(yb["training_labels"], yr["training_labels"])
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    panels = [
        (xb["training"], None, "(a) Training data with\ntarget corruption", "#d62728"),
        (xb["generated"], None, "(b) OT-FM", "#1f77b4"),
        (xr["generated"], None, "(c) FM-TRS", "#1f77b4"),
        (
            yb["training"],
            yb["training_labels"],
            "(d) Training data with\nlabel noise",
            None,
        ),
        (yb["generated"], yb["generated_labels"], "(e) OT-FM", None),
        (yr["generated"], yr["generated_labels"], "(f) FM-CCS", None),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "text.color": "#151515",
            "axes.labelcolor": "#151515",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    limits = []
    for group in range(2):
        # Common limits within each corruption experiment, including ALL points.
        all_points = np.concatenate([panels[group * 3 + j][0] for j in range(3)])
        if not np.isfinite(all_points).all():
            raise ValueError(
                "Non-finite samples found; inspect training before plotting."
            )
        low, high = all_points.min(axis=0), all_points.max(axis=0)
        center = (low + high) / 2
        half_span = max(float(np.max(high - low)), 1.0) * 0.56
        limits.append((center - half_span, center + half_span))

    short_titles = [
        "(a) Training data",
        "(b) OT-FM",
        "(c) FM-TRS",
        "(d) Training data",
        "(e) OT-FM",
        "(f) FM-CCS",
    ]

    def draw_panel(ax, i, standalone=False):
        points, labels, title, color = panels[i]
        pts, labs = subset(points, n_plot, manifest["seed"] + 777, labels)
        colors = color if labs is None else np.asarray(COLORS)[labs]
        low, high = limits[i // 3]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=colors,
            s=4,
            alpha=0.65,
            edgecolors="none",
            rasterized=True,
        )
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect="equal")
        if standalone and i in (0, 3):
            title = (
                "(a) Target-corrupted\ntraining data"
                if i == 0
                else "(d) Label-noisy\ntraining data"
            )
        else:
            title = short_titles[i]
        ax.set_title(title, fontsize=18, fontweight="medium", pad=6)
        ax.grid(alpha=0.10, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=18, length=5, width=1, pad=5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#555555")

    if layout == 'row':
        fig = plt.figure(figsize=(24.0, 5.8))
        groups = fig.add_gridspec(1, 2, left=0.035, right=0.992,
                                  bottom=0.21, top=0.74, wspace=0.14)
        axes = []
        for group in range(2):
            panels_grid = groups[0, group].subgridspec(1, 3, wspace=0.25)
            axes.extend(fig.add_subplot(panels_grid[0, j]) for j in range(3))
        axes = np.asarray(axes)
    else:
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 11.0))
    for i, ax in enumerate(axes.flat):
        draw_panel(ax, i)
    handles = [Line2D([], [], color=c, marker='o', linestyle='', markersize=9,
                      label=f'Class {i}') for i, c in enumerate(COLORS)]
    if manifest['smoke_test']:
        fig.suptitle('SMOKE TEST ONLY - 8 training steps', color='red', fontsize=20)
    if layout != 'row':
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.86,
                             wspace=0.30, hspace=0.35)
    # Keep class colors with the conditional panels only. Two short legend rows
    # preserve readable labels without adding headings above the plots.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes.flat) / fig.bbox.height
    legend_gap = 4.0 / (72.0 * fig.get_figheight())
    legend_style = dict(loc='upper center', frameon=False, fontsize=18,
                        handletextpad=0.25, columnspacing=1.0,
                        borderaxespad=0, borderpad=0, labelspacing=0.35)
    legend_top = axes_bottom - legend_gap
    if layout == 'row':
        left_center = (axes[0].get_position().x0 + axes[2].get_position().x1) / 2
        right_center = (axes[3].get_position().x0 + axes[5].get_position().x1) / 2
        sample_handles = [
            Line2D([], [], color=color, marker='o', linestyle='', markersize=9,
                   label=label)
            for color, label in [('#d62728', 'Training data'),
                                 ('#1f77b4', 'Generated samples')]]
        sample_legend = fig.legend(handles=sample_handles, ncol=2,
                                   title='No class labels', title_fontsize=18,
                                   bbox_to_anchor=(left_center, legend_top), **legend_style)
        # Matplotlib fills legend columns first; reorder for rows 0--4 and 5--9.
        class_handles = [handles[i] for pair in zip(range(5), range(5, 10)) for i in pair]
        class_legend = fig.legend(handles=class_handles, ncol=5,
                                  bbox_to_anchor=(right_center, legend_top), **legend_style)
        fig.canvas.draw()
        # Center the divider in the visible gap, accounting for panel (d)'s ticks.
        divider_x = (axes[2].get_tightbbox(renderer).x1 +
                     axes[3].get_tightbbox(renderer).x0) / (2 * fig.bbox.width)
        divider_bottom = min(legend.get_window_extent(renderer).y0
                             for legend in (sample_legend, class_legend)) / fig.bbox.height
        divider_top = max(ax.get_tightbbox(renderer).y1 for ax in axes) / fig.bbox.height
        fig.add_artist(Line2D([divider_x, divider_x], [divider_bottom, divider_top],
                              transform=fig.transFigure, color='#b0b0b0', linewidth=1.0))
    else:
        fig.legend(handles=handles, ncol=5,
                   bbox_to_anchor=(0.5, legend_top), **legend_style)
    figures = out / "Figures"
    figures.mkdir(exist_ok=True)
    save_figure(fig, figures / figure_name)
    plt.close(fig)

    panel_dir = figures / (
        "panels" if figure_name == "spirals_figure_y_v2" else f"{figure_name}_panels"
    )
    panel_dir.mkdir(exist_ok=True)
    names = [
        "a_target_training",
        "b_target_ot_fm",
        "c_target_fm_trs",
        "d_label_training",
        "e_label_ot_fm",
        "f_label_fm_ccs",
    ]
    for i, name in enumerate(names):
        panel_fig, ax = plt.subplots(figsize=(4.4, 4.9))
        panel_fig.subplots_adjust(left=0.17, right=0.96, bottom=0.14, top=0.77)
        draw_panel(ax, i, standalone=True)
        if manifest["smoke_test"]:
            panel_fig.suptitle("SMOKE TEST ONLY", color="red", fontsize=14)
        # Fixed-size canvases make the six panels easy to align in PowerPoint.
        save_figure(panel_fig, panel_dir / name, tight=False)
        plt.close(panel_fig)
    summary = {
        "target_ot_fm": xb_result,
        "target_fm_trs": xr_result,
        "label_ot_fm": yb_result,
        "label_fm_ccs": yr_result,
    }
    (out / "figure_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
