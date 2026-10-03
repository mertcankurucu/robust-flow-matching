"""Run one core 2D experiment."""

import argparse
from .train import Config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["target", "label"], default="target")
    parser.add_argument(
        "--method", choices=["ot_fm", "fm_trs", "fm_ccs"], default="ot_fm"
    )
    parser.add_argument("--dataset", choices=["moons", "2spirals", "segmented_spirals"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10001)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--pool-size", type=int, default=50000)
    parser.add_argument("--generated", type=int)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument(
        "--noise-type", choices=["symmetric", "adjacent"], default="symmetric"
    )
    args = vars(parser.parse_args())
    output = args.pop("output_dir")
    is_label = args["mode"] == "label"
    for key, value in dict(
        dataset="segmented_spirals" if is_label else "2spirals",
        generated=5000 if is_label else 50000,
        tau=0.01 if is_label else 0.001,
        rate=0.4 if is_label else 0.2,
    ).items():
        if args[key] is None:
            args[key] = value
    run(Config(**args), output)


if __name__ == "__main__":
    main()
