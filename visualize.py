#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")


def parse_filename(stem):
    return dict(p.split("=") for p in stem.split("_"))


def thread_label(v):
    """Normalise thread field: int string -> 'Nt', legacy strings kept as-is."""
    try:
        return f"{int(v)}t"
    except ValueError:
        return v  # 'single', 'multi' from old files


def config_key(row):
    return f"{row['alloc']} ({thread_label(row['thread'])})"


def load_all():
    train_dfs, infer_dfs = [], []
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        try:
            meta = parse_filename(f.stem)
            df = pd.read_csv(f)
            for k, v in meta.items():
                df[k] = v
            (train_dfs if meta.get("mode") == "train" else infer_dfs).append(df)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")
    train = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    infer = pd.concat(infer_dfs, ignore_index=True) if infer_dfs else pd.DataFrame()
    if not train.empty:
        train["config"] = train.apply(config_key, axis=1)
    if not infer.empty:
        infer["config"] = infer.apply(config_key, axis=1)
    return train, infer


def _color_map(keys):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {k: colors[i % len(colors)] for i, k in enumerate(sorted(keys))}


def plot_training(train):
    activations = sorted(train["act"].unique())
    configs = sorted(train["config"].unique())
    cmap = _color_map(configs)
    n_acts = len(activations)

    fig, axes = plt.subplots(n_acts, 3, figsize=(15, 4 * n_acts), squeeze=False)
    fig.suptitle("Training", fontsize=14, fontweight="bold")

    for row, act in enumerate(activations):
        ax_loss, ax_acc, ax_spd = axes[row]
        act_data = train[train["act"] == act]
        runs_seen = {cfg: 0 for cfg in configs}

        for cfg in configs:
            cfg_data = act_data[act_data["config"] == cfg]
            n_runs = cfg_data["run"].nunique()
            for run in sorted(cfg_data["run"].unique()):
                runs_seen[cfg] += 1
                run_data = cfg_data[cfg_data["run"] == run].sort_values("epoch")
                color = cmap[cfg]
                alpha = 0.3 + 0.7 * (runs_seen[cfg] / n_runs)

                ax_loss.plot(run_data["epoch"], run_data["loss"], color=color, alpha=alpha)
                acc = run_data["correct"] / run_data["test_size"]
                ax_acc.plot(run_data["epoch"], acc, color=color, alpha=alpha)
                ms = run_data["elapsed_ns"] / 1e6
                ax_spd.plot(run_data["epoch"], ms, color=color, alpha=alpha)

        ax_loss.set(title=f"{act} — Loss", xlabel="Epoch", ylabel="Loss")
        ax_acc.set(title=f"{act} — Accuracy", xlabel="Epoch", ylabel="correct / total")
        ax_spd.set(title=f"{act} — Epoch Time", xlabel="Epoch", ylabel="ms")

    handles = [mlines.Line2D([], [], color=cmap[c], label=c) for c in configs]
    fig.legend(handles=handles, loc="upper right", title="Config")
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    return fig


def plot_inference(infer):
    infer = infer.copy()
    infer["accuracy"] = infer["correct"] / infer["test_size"]
    infer["throughput"] = infer["test_size"] / (infer["elapsed_ns"] / 1e9)

    activations = sorted(infer["act"].unique())
    configs = sorted(infer["config"].unique())
    cmap = _color_map(configs)

    fig, (ax_acc, ax_thr) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Inference", fontsize=14, fontweight="bold")

    n = len(configs)
    width = 0.7 / n
    xs = range(len(activations))

    for i, cfg in enumerate(configs):
        cfg_data = infer[infer["config"] == cfg]
        offset = (i - n / 2 + 0.5) * width
        accs = [cfg_data[cfg_data["act"] == a]["accuracy"].mean() for a in activations]
        thrs = [cfg_data[cfg_data["act"] == a]["throughput"].mean() for a in activations]
        ax_acc.bar([x + offset for x in xs], accs, width, label=cfg, color=cmap[cfg])
        ax_thr.bar([x + offset for x in xs], thrs, width, label=cfg, color=cmap[cfg])

    for ax, ylabel, title in [
        (ax_acc, "correct / total", "Accuracy"),
        (ax_thr, "samples / sec", "Throughput"),
    ]:
        ax.set(title=title, ylabel=ylabel, xticks=list(xs), xticklabels=activations)
        ax.legend(title="Config")

    fig.tight_layout()
    return fig


def main():
    if not RESULTS_DIR.exists():
        print(f"Results dir not found: {RESULTS_DIR}")
        sys.exit(1)

    train, infer = load_all()

    if train.empty and infer.empty:
        print("No data found.")
        sys.exit(1)

    if not train.empty:
        runs = train["run"].nunique()
        print(f"Training: {len(train)} rows, {runs} run(s), "
              f"acts={sorted(train['act'].unique())}, configs={sorted(train['config'].unique())}")
        plot_training(train)

    if not infer.empty:
        runs = infer["run"].nunique()
        print(f"Inference: {len(infer)} rows, {runs} run(s)")
        plot_inference(infer)

    plt.show()


if __name__ == "__main__":
    main()
