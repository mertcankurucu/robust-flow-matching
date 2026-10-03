"""Collect measured results from the nine core method/condition runs."""

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path

METRICS = (
    "w2_clean",
    "mmd",
    "coverage",
    "density",
    "cw_w2",
    "cw_coverage",
    "fid",
    "is_mean",
    "is_std",
    "cw_fid",
    "cas",
)


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def read_rows(root):
    cases = [
        ("2d", "target", "ot_fm"),
        ("2d", "target", "fm_trs"),
        ("2d", "label", "ot_fm"),
        ("2d", "label", "fm_ccs"),
        ("cifar10", "target", "ot_fm"),
        ("cifar10", "target", "fm_trs"),
        ("cifar10", "label", "ot_fm"),
        ("cifar10", "label", "fm_ccs_tau0.001"),
        ("cifar10", "label", "fm_ccs_tau0.01"),
    ]
    rows = []
    for dataset, noise, method in cases:
        if dataset == "2d":
            group = "2d_xnoise" if noise == "target" else "2d_ynoise"
            path = root / "2d" / group / method / "result.json"
        else:
            phase = "evaluate" if noise == "label" else "evaluation"
            path = root / f"cifar_{noise}_{method}" / phase / "metrics.json"
        row = dict(
            dataset=dataset,
            noise=noise,
            method=method,
            status="MISSING",
            source=str(path.relative_to(root)),
        )
        rows.append(row)
        if not path.exists():
            continue
        try:
            payload = read(path)
            if dataset == "2d":
                cfg, metrics = payload["config"], payload["metrics"]
                required = (
                    ("w2_clean", "mmd", "coverage", "density")
                    if noise == "target"
                    else ("cw_w2", "cw_coverage")
                )
                manifest = read(root / "2d/manifest.json")
                if manifest.get("smoke_test"):
                    raise ValueError("A diagnostic run is not a full result")
            elif noise == "target":
                cfg = read(path.parent.parent / "config.json")
                metrics = payload
                required = ("fid", "is_mean", "is_std", "coverage", "density")
                log = path.parent.parent / "train.jsonl"
                records = [
                    json.loads(line)
                    for line in log.read_text().splitlines()
                    if line.strip()
                ]
                if not records or records[-1]["epoch"] + 1 != cfg["epochs"]:
                    raise ValueError(
                        "Training has not reached its configured final epoch"
                    )
            else:
                case = path.parent.parent
                if not (path.parent / "completed.json").is_file():
                    row["status"] = "INCOMPLETE"
                    continue
                cfg = read(case / "train/resolved_config.json")
                records = [
                    json.loads(line)
                    for line in (path.parent / "log.txt").read_text().splitlines()
                    if line.strip()
                ]
                final = records[-1]
                # Evaluation logs the checkpoint epoch + 1; only the final checkpoint counts.
                if final.get("epoch") != cfg["epochs"]:
                    raise ValueError(
                        "Training has not reached its configured final epoch"
                    )
                metrics = {
                    key.removeprefix("eval_"): value
                    for key, value in final.items()
                    if key.startswith("eval_")
                }
                metrics.update(
                    cw_fid=payload["overall"]["cwfid_mean"],
                    cw_coverage=payload["overall"]["cw_coverage"],
                    per_class=payload["per_class"],
                )
                if len(payload["per_class"]) != 10:
                    raise ValueError("Expected all ten class-wise results")
                for values in payload["per_class"].values():
                    for key in ("cwfid", "coverage"):
                        if not math.isfinite(float(values[key])):
                            raise ValueError(f"Invalid per-class {key}")
                if "cas" in payload["overall"]:
                    metrics["cas"] = payload["overall"]["cas"]
                required = ("fid", "is_mean", "cw_fid", "cw_coverage")
            if cfg.get("smoke") or metrics.get("smoke_test"):
                raise ValueError("A diagnostic run is not a full result")
            for key in required:
                if not math.isfinite(float(metrics[key])):
                    raise ValueError(f"Invalid metric: {key}")
            row.update(
                status="COMPLETE",
                seed=cfg["seed"],
                epochs=cfg.get("epochs"),
                steps=cfg.get("steps"),
            )
            row.update({key: metrics[key] for key in METRICS if key in metrics})
            row["per_class"] = metrics.get("per_class", {})
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
        ) as error:
            row.update(status="INVALID", error=str(error))
    return rows


def table(rows, columns):
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            cells.append(
                "-"
                if value is None
                else f"{value:.6g}"
                if isinstance(value, float)
                else str(value)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def atomic_write(path, contents):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"Missing results directory: {args.root}")
    rows = read_rows(args.root)
    complete = sum(row["status"] == "COMPLETE" for row in rows)
    common = [("noise", "Noise"), ("method", "Method")]
    text = f"# Core results: {complete}/9 runs complete\n\nMeasured results; missing values are never replaced with zero.\n\n"
    text += "## 2D\n\n" + table(
        rows[:4],
        common
        + [
            ("w2_clean", "W2"),
            ("mmd", "MMD"),
            ("coverage", "Coverage"),
            ("density", "Density"),
            ("cw_w2", "CW-W2"),
            ("cw_coverage", "CW-Coverage"),
            ("status", "Status"),
        ],
    )
    text += "\n\n## CIFAR-10\n\n" + table(
        rows[4:],
        common
        + [
            ("epochs", "Epochs"),
            ("fid", "FID"),
            ("is_mean", "IS"),
            ("is_std", "IS std"),
            ("coverage", "Coverage"),
            ("density", "Density"),
            ("cw_fid", "CW-FID"),
            ("cas", "CAS"),
            ("cw_coverage", "CW-Coverage"),
            ("status", "Status"),
        ],
    )
    text += "\n\nCAS requires a supplied CIFAR-10 classifier. CSV/JSON retain numeric precision.\n"
    for row in rows:
        if "error" in row:
            text += f"\n{row['source']}: {row['error']}\n"
    atomic_write(args.root / "summary.md", text)
    atomic_write(
        args.root / "summary.json", json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "dataset",
            "noise",
            "method",
            "status",
            "seed",
            "epochs",
            "steps",
            *METRICS,
            "source",
            "error",
        ],
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(args.root / "summary.csv", buffer.getvalue())
    print(text, flush=True)
    return 0 if complete == 9 or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
