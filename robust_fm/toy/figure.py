"""Train the four models for the paper's compact, single-row spiral figure."""

import argparse
import json
from pathlib import Path

from .plot import plot_figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=10001)
    parser.add_argument("--n-plot", type=int, default=4000)
    parser.add_argument("--figure-name", default="spirals_figure_y_v2")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="8-step pipeline check; not scientific results",
    )
    args = parser.parse_args()
    if (
        args.n_plot < 1
        or not args.figure_name
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in args.figure_name
        )
    ):
        parser.error("Use a positive plot count and a simple filename stem")
    out = args.output_dir.resolve()
    if not args.plot_only:
        import torch
        from .train import Config, run

        if out.exists() and any(out.iterdir()):
            parser.error(
                "Output must be new/empty; use --plot-only for existing results"
            )
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "smoke_test": args.smoke,
                    "distance": args.distance,
                    "description": "Latest core defaults; compact one-row figure",
                },
                indent=2,
            ),
            encoding="utf8",
        )
        if args.smoke:
            torch.set_num_threads(2)
        for mode, method in [
            ("target", "ot_fm"),
            ("target", "fm_trs"),
            ("label", "ot_fm"),
            ("label", "fm_ccs"),
        ]:
            cfg = Config(
                mode=mode,
                method=method,
                seed=args.seed,
                device=args.device,
                steps=args.steps,
                distance=args.distance,
                dataset="2spirals" if mode == "target" else "segmented_spirals",
                tau=0.001 if mode == "target" else 0.01,
                rate=0.2 if mode == "target" else 0.4,
                generated=50000 if mode == "target" else 5000,
            )
            if args.smoke:
                cfg.steps, cfg.batch_size, cfg.hidden_dim = 8, 32, 32
                cfg.pool_size, cfg.generated, cfg.score_freeze_steps, cfg.max_ot = (
                    100,
                    100,
                    2,
                    100,
                )
            run(cfg, out / ("2d_xnoise" if mode == "target" else "2d_ynoise") / method)
    plot_figure(out, args.n_plot, layout="row", figure_name=args.figure_name)


if __name__ == "__main__":
    main()
