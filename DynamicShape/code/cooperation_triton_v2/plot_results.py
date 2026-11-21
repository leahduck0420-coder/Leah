"""Render publication-ready experiment charts with Matplotlib on a headless server.

The module reads only the CSV files produced by :mod:`report`.  It can therefore
be rerun without compiling kernels or repeating the GPU benchmark::

    python -m cooperation_triton_v2.plot_results --run results/v2_targeted
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm


OPS = ("gemm", "linear", "linear_post", "qkt", "pv", "moe")
OP_NAMES = {
    "gemm": "GEMM",
    "linear": "Linear",
    "linear_post": "Linear + post-op",
    "qkt": "Attention QK^T",
    "pv": "Attention PV",
    "moe": "MoE expert GEMM",
}
METHODS = (
    "inductor_default",
    "triton_padding_autotune",
    "triton_padding_bound",
    "model_bound",
    "optimized_bound",
    "optimized_dispatch",
    "optimized_guarded",
)
METHOD_NAMES = {
    "inductor_default": "Inductor default",
    "triton_padding_autotune": "Triton padding autotune",
    "triton_padding_bound": "Triton padding bound",
    "model_bound": "Local model",
    "optimized_bound": "Policy-table kernel",
    "optimized_dispatch": "Policy table + dispatch",
    "optimized_guarded": "Full guarded call",
}
METHOD_COLORS = {
    "inductor_default": "#4c78a8",
    "triton_padding_autotune": "#f58518",
    "triton_padding_bound": "#eeca3b",
    "model_bound": "#b279a2",
    "optimized_bound": "#54a24b",
    "optimized_dispatch": "#2a9d8f",
    "optimized_guarded": "#e45756",
}
POLICIES = ("padding", "small_mma", "hybrid_k", "hybrid")
POLICY_COLORS = {
    "padding": "#4c78a8",
    "small_mma": "#f58518",
    "hybrid_k": "#54a24b",
    "hybrid": "#e45756",
}
KIND_ORDER = {
    "parent_aligned_control": 0,
    "m_tail": 1,
    "n_tail": 2,
    "k_tail_1": 3,
    "k_tail_3": 4,
    "mixed_tail": 5,
    "negative_control": 6,
}
KIND_SHORT = {
    "parent_aligned_control": "aligned",
    "m_tail": "M+1",
    "n_tail": "N+1",
    "k_tail_1": "K+1",
    "k_tail_3": "K+3",
    "mixed_tail": "mixed",
    "negative_control": "negative",
}


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _number(value, default=math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _case_sort(row: dict[str, str]):
    op = OPS.index(row["op"]) if row["op"] in OPS else len(OPS)
    return (
        op,
        row.get("pair", ""),
        KIND_ORDER.get(row.get("kind", ""), 99),
        int(row.get("m", 0)),
        int(row.get("n", 0)),
        int(row.get("k", 0)),
        row.get("case_id", ""),
    )


def _cases(rows: list[dict[str, str]], op: str) -> list[dict[str, str]]:
    unique = {}
    for row in rows:
        if row["op"] == op:
            unique.setdefault(row["case_id"], row)
    return sorted(unique.values(), key=_case_sort)


def _label(row: dict[str, str]) -> str:
    return f'{row["shape"]}\n{KIND_SHORT.get(row.get("kind", ""), row.get("kind", ""))}'


def _lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["case_id"], row["method"]): row for row in rows}


def _speed(index, case_id: str, method: str, baseline: str, metric="request_us"):
    target = index.get((case_id, method))
    base = index.get((case_id, baseline))
    if not target or not base:
        return math.nan
    target_value = _number(target.get(metric))
    base_value = _number(base.get(metric))
    return base_value / target_value if target_value > 0 and base_value > 0 else math.nan


def _footer(fig, text: str):
    fig.text(0.01, 0.006, text, ha="left", va="bottom", fontsize=8, color="#4b5563")


def _save(fig, directory: Path, stem: str, formats: tuple[str, ...], manifest: list[dict], description: str):
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 180
        fig.savefig(path, **kwargs)
        files.append(path.name)
    plt.close(fig)
    manifest.append({"id": stem, "description": description, "files": files})


def _plot_speedup_heatmap(rows, directory, formats, manifest):
    index = _lookup(rows)
    ops = [op for op in OPS if _cases(rows, op)]
    baselines = ("inductor_default", "triton_padding_autotune")
    values = []
    for op in ops:
        for baseline in baselines:
            values.extend(
                _speed(index, case["case_id"], "optimized_guarded", baseline)
                for case in _cases(rows, op)
            )
    finite = [value for value in values if math.isfinite(value)]
    low = min([0.85, *finite]) if finite else 0.85
    high = max([1.15, *finite]) if finite else 1.15
    low = max(0.5, low)
    high = min(2.0, high)
    if not low < 1 < high:
        low, high = 0.85, 1.15
    norm = TwoSlopeNorm(vmin=low, vcenter=1.0, vmax=high)
    fig, axes = plt.subplots(len(ops), 2, figsize=(22, max(4, 2.25 * len(ops))), squeeze=False)
    image = None
    for row_no, op in enumerate(ops):
        cases = _cases(rows, op)
        labels = [_label(case) for case in cases]
        for col_no, baseline in enumerate(baselines):
            axis = axes[row_no, col_no]
            data = np.array([[_speed(index, case["case_id"], "optimized_guarded", baseline) for case in cases]])
            masked = np.ma.masked_invalid(data)
            image = axis.imshow(masked, cmap="RdYlGn", norm=norm, aspect="auto")
            axis.set_yticks([0], [OP_NAMES.get(op, op)])
            axis.set_xticks(range(len(labels)), labels, rotation=48, ha="right", fontsize=7)
            if row_no == 0:
                axis.set_title(f'Full guarded call / {METHOD_NAMES[baseline]}')
            for x, value in enumerate(data[0]):
                if math.isfinite(value):
                    axis.text(x, 0, f"{value:.2f}x", ha="center", va="center", fontsize=6,
                              color="black" if 0.72 < norm(value) < 0.9 else "white")
            axis.set_xlabel("shape and tail kind")
    if image is not None:
        fig.colorbar(image, ax=axes, pad=0.008, fraction=0.018, label="Speedup (>1 is faster)")
    fig.suptitle("Speedup heatmap for the final guarded strategy", fontsize=16, y=1.015)
    _footer(fig, "Median held-out request latency. Missing or failed measurements remain blank; values are not imputed.")
    fig.subplots_adjust(hspace=1.0, wspace=0.12, bottom=0.12, top=0.90)
    _save(fig, directory, "01_speedup_heatmap", formats, manifest,
          "Final guarded speedup for every operator and shape against Inductor and Triton padding autotune.")


def _plot_latency(rows, directory, formats, manifest):
    index = _lookup(rows)
    for op in OPS:
        cases = _cases(rows, op)
        if not cases:
            continue
        labels = [_label(case) for case in cases]
        fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        for axis, metric, title in zip(
            axes,
            ("request_us", "device_span_us"),
            ("End-to-end request latency", "GPU device span"),
        ):
            for method in METHODS:
                series = []
                for case in cases:
                    row = index.get((case["case_id"], method))
                    series.append(_number(row.get(metric)) if row else math.nan)
                if np.isfinite(series).any():
                    axis.plot(range(len(cases)), series, marker="o", markersize=3.5, linewidth=1.4,
                              label=METHOD_NAMES[method], color=METHOD_COLORS[method])
            axis.set_yscale("log")
            axis.set_ylabel("Latency (us, log scale)")
            axis.set_title(title)
            axis.grid(True, which="both", axis="y", alpha=0.25)
        axes[-1].set_xticks(range(len(labels)), labels, rotation=50, ha="right", fontsize=8)
        axes[-1].set_xlabel("shape and tail kind")
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96))
        fig.suptitle(f'{OP_NAMES.get(op, op)}: compiler baselines and optimization paths', fontsize=16)
        _footer(fig, "Request latency includes Python dispatch and synchronization; device span covers recorded GPU work and feed gaps.")
        fig.subplots_adjust(top=0.89, bottom=0.18, hspace=0.16)
        _save(fig, directory, f"02_latency_{op}", formats, manifest,
              f"Request and device latency for all compiler, model, table, dispatch and guarded paths of {op}.")


def _plot_ablation(rows, directory, formats, manifest):
    index = _lookup(rows)
    tiles = sorted({row["tile"] for row in rows if row.get("policy") in POLICIES and row.get("tile")})
    for tile in tiles:
        ops = [op for op in OPS if _cases(rows, op)]
        fig, axes = plt.subplots(len(ops), 1, figsize=(19, max(6, 3.0 * len(ops))), squeeze=False)
        plotted = False
        for row_no, op in enumerate(ops):
            axis = axes[row_no, 0]
            cases = _cases(rows, op)
            labels = [_label(case) for case in cases]
            pad_method = f"padding_{tile}"
            for policy in POLICIES:
                method = f"{policy}_{tile}"
                series = [_speed(index, case["case_id"], method, pad_method) for case in cases]
                if np.isfinite(series).any():
                    plotted = True
                    axis.plot(range(len(cases)), series, marker="o", markersize=3.5, linewidth=1.3,
                              color=POLICY_COLORS[policy], label=policy)
            axis.axhline(1.0, color="#111827", linewidth=1, linestyle="--")
            axis.set_ylabel(f"{OP_NAMES.get(op, op)}\nspeedup")
            axis.grid(True, axis="y", alpha=0.25)
            axis.set_xticks(range(len(labels)), labels, rotation=48, ha="right", fontsize=7)
        if not plotted:
            plt.close(fig)
            continue
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if not handles:
            for axis in axes[:, 0]:
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    break
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97))
        fig.suptitle(f"Fixed-tile policy ablation: {tile}", fontsize=16)
        _footer(fig, "Each point is padding latency divided by policy latency at the same Tile; >1 means the policy is faster. Gaps are ineligible policies.")
        fig.subplots_adjust(top=0.93, bottom=0.08, hspace=0.95)
        _save(fig, directory, f"03_ablation_tile_{tile}", formats, manifest,
              f"Padding, small_mma, hybrid_k and hybrid comparison at fixed Tile {tile}.")


def _plot_strategy(rows, directory, formats, manifest):
    index = _lookup(rows)
    ops = [op for op in OPS if _cases(rows, op)]
    cmap = ListedColormap([POLICY_COLORS[p] for p in POLICIES])
    norm = BoundaryNorm(np.arange(-0.5, len(POLICIES) + 0.5), cmap.N)
    fig, axes = plt.subplots(len(ops), 1, figsize=(20, max(5, 2.1 * len(ops))), squeeze=False)
    counts = Counter()
    for row_no, op in enumerate(ops):
        axis = axes[row_no, 0]
        cases = _cases(rows, op)
        values, tiles, labels = [], [], []
        for case in cases:
            selected = index.get((case["case_id"], "optimized_bound"))
            policy = selected.get("policy", "") if selected else ""
            values.append(POLICIES.index(policy) if policy in POLICIES else math.nan)
            tiles.append(selected.get("tile", "") if selected else "")
            labels.append(_label(case))
            if policy:
                counts[policy] += 1
        data = np.ma.masked_invalid(np.array([values], dtype=float))
        axis.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        axis.set_yticks([0], [OP_NAMES.get(op, op)])
        axis.set_xticks(range(len(labels)), labels, rotation=48, ha="right", fontsize=7)
        for x, tile in enumerate(tiles):
            if tile:
                axis.text(x, 0, tile.replace("x", "×"), ha="center", va="center", fontsize=6, color="white")
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", markersize=10,
                          markerfacecolor=POLICY_COLORS[p], markeredgecolor="none",
                          label=f"{p} (n={counts[p]})") for p in POLICIES]
    fig.suptitle("Policy-table selection by operator and shape", fontsize=16, y=1.01)
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.975))
    _footer(fig, "Cell text is BM×BN×BK. The selection is frozen after discovery; the final guard may route the call back to a compiler baseline.")
    fig.subplots_adjust(top=0.88, bottom=0.13, hspace=1.0)
    _save(fig, directory, "04_strategy_selection", formats, manifest,
          "Selected policy and Tile for every operator and shape.")


def _summary_values(rows, baseline):
    index = _lookup(rows)
    result = {}
    for op in OPS:
        values = [_speed(index, case["case_id"], "optimized_guarded", baseline) for case in _cases(rows, op)]
        values = [value for value in values if math.isfinite(value) and value > 0]
        if values:
            geomean = math.exp(sum(math.log(value) for value in values) / len(values))
            result[op] = (geomean, min(values), max(values), values)
    return result


def _plot_summary(rows, directory, formats, manifest):
    baselines = ("inductor_default", "triton_padding_autotune")
    data = {baseline: _summary_values(rows, baseline) for baseline in baselines}
    ops = [op for op in OPS if any(op in data[baseline] for baseline in baselines)]
    x = np.arange(len(ops), dtype=float)
    width = 0.34
    fig, axis = plt.subplots(figsize=(13, 7))
    colors = (METHOD_COLORS["inductor_default"], METHOD_COLORS["triton_padding_autotune"])
    for number, (baseline, color) in enumerate(zip(baselines, colors)):
        centers, lower, upper = [], [], []
        for op in ops:
            geomean, worst, best, _ = data[baseline].get(op, (math.nan,) * 4)
            centers.append(geomean)
            lower.append(geomean - worst if math.isfinite(geomean) else 0)
            upper.append(best - geomean if math.isfinite(geomean) else 0)
        positions = x + (number - 0.5) * width
        bars = axis.bar(positions, centers, width, color=color, alpha=0.85,
                        label=f'vs {METHOD_NAMES[baseline]}')
        axis.errorbar(positions, centers, yerr=np.array([lower, upper]), fmt="none", ecolor="#111827",
                      capsize=4, linewidth=1)
        for bar, center in zip(bars, centers):
            if math.isfinite(center):
                axis.text(bar.get_x() + bar.get_width() / 2, center, f"{center:.2f}x",
                          ha="center", va="bottom", fontsize=8)
    axis.axhline(1.0, color="#111827", linestyle="--", linewidth=1)
    axis.set_xticks(x, [OP_NAMES.get(op, op) for op in ops], rotation=20, ha="right")
    axis.set_ylabel("Speedup (>1 is faster)")
    axis.set_title("Benefit and regression summary by operator")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False)
    _footer(fig, "Bar: geometric mean. Error range: worst to best held-out shape. The range is descriptive, not a confidence interval.")
    fig.subplots_adjust(bottom=0.19)
    _save(fig, directory, "05_benefit_regression_summary", formats, manifest,
          "Geometric mean, best and worst final speedup by operator against both requested baselines.")

    fig, axis = plt.subplots(figsize=(14, 8))
    labels, wins, neutral, regressions = [], [], [], []
    for op in ops:
        for baseline in baselines:
            values = data[baseline].get(op, (0, 0, 0, []))[3]
            labels.append(f'{OP_NAMES.get(op, op)}\nvs {"Inductor" if baseline == "inductor_default" else "Triton auto"}')
            wins.append(sum(value > 1.05 for value in values))
            regressions.append(sum(value < 1 / 1.05 for value in values))
            neutral.append(len(values) - wins[-1] - regressions[-1])
    positions = np.arange(len(labels))
    axis.bar(positions, wins, color="#54a24b", label=">5% faster")
    axis.bar(positions, neutral, bottom=wins, color="#bab0ac", label="within ±5%")
    axis.bar(positions, regressions, bottom=np.array(wins) + np.array(neutral), color="#e45756", label=">5% slower")
    axis.set_xticks(positions, labels, rotation=38, ha="right")
    axis.set_ylabel("Number of held-out shapes")
    axis.set_title("Wins, neutral cases and regressions")
    axis.legend(frameon=False, ncol=3)
    axis.grid(True, axis="y", alpha=0.2)
    _footer(fig, "Classification uses the full guarded call and median request latency. Missing cases are excluded rather than counted as neutral.")
    fig.subplots_adjust(bottom=0.25)
    _save(fig, directory, "05b_win_regression_counts", formats, manifest,
          "Counts of meaningful wins, neutral cases and regressions for each operator and baseline.")


def _plot_tail_pairs(pairs, directory, formats, manifest):
    if not pairs:
        return
    ops = [op for op in OPS if any(row["op"] == op for row in pairs)]
    fig, axes = plt.subplots(len(ops), 1, figsize=(17, max(6, 3.2 * len(ops))), squeeze=False)
    for row_no, op in enumerate(ops):
        axis = axes[row_no, 0]
        subset = sorted((row for row in pairs if row["op"] == op), key=_case_sort)
        labels = [f'{row["shape"]}\n{KIND_SHORT.get(row.get("kind", ""), row.get("kind", ""))}' for row in subset]
        for key, label, color in (
            ("inductor_tail_ratio", "Inductor tail/aligned", METHOD_COLORS["inductor_default"]),
            ("triton_padding_tail_ratio", "Triton padding tail/aligned", METHOD_COLORS["triton_padding_bound"]),
            ("final_tail_ratio", "Full guarded tail/aligned", METHOD_COLORS["optimized_guarded"]),
        ):
            axis.plot(range(len(subset)), [_number(row.get(key)) for row in subset], marker="o",
                      linewidth=1.4, markersize=4, label=label, color=color)
        axis.axhline(1.0, color="#111827", linestyle="--", linewidth=1)
        axis.set_ylabel(f"{OP_NAMES.get(op, op)}\ntail/aligned")
        axis.set_xticks(range(len(labels)), labels, rotation=48, ha="right", fontsize=7)
        axis.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("Tail-shape cost relative to its aligned parent", fontsize=16)
    _footer(fig, "Ratio uses median request latency for each tail shape divided by the paired aligned control; lower is better.")
    fig.subplots_adjust(top=0.92, bottom=0.09, hspace=1.0)
    _save(fig, directory, "06_tail_pair_comparison", formats, manifest,
          "Paired tail-to-aligned latency ratios for Inductor, Triton padding and the full guarded strategy.")


def generate_plots(run_dir, formats=("png", "svg", "pdf")):
    """Generate every chart in ``run_dir/figures`` and return the manifest."""
    run_dir = Path(run_dir)
    formats = tuple(extension.lower().lstrip(".") for extension in formats)
    invalid = set(formats) - {"png", "svg", "pdf"}
    if invalid:
        raise ValueError(f"unsupported formats: {sorted(invalid)}")
    rows = _read(run_dir / "comparison.csv")
    pairs = _read(run_dir / "tail_pairs.csv")
    if not rows:
        raise ValueError(f"no rows found in {run_dir / 'comparison.csv'}")
    directory = run_dir / "figures"
    manifest = []
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    _plot_speedup_heatmap(rows, directory, formats, manifest)
    _plot_latency(rows, directory, formats, manifest)
    _plot_ablation(rows, directory, formats, manifest)
    _plot_strategy(rows, directory, formats, manifest)
    _plot_summary(rows, directory, formats, manifest)
    _plot_tail_pairs(pairs, directory, formats, manifest)
    payload = {
        "schema": 1,
        "backend": "matplotlib-agg",
        "source": ["comparison.csv", "before_after.csv", "domain_summary.csv", "tail_pairs.csv"],
        "formats": list(formats),
        "plots": manifest,
    }
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Benchmark result directory containing comparison.csv")
    parser.add_argument("--formats", nargs="+", default=("png", "svg", "pdf"),
                        choices=("png", "svg", "pdf"))
    args = parser.parse_args()
    manifest = generate_plots(args.run, tuple(args.formats))
    print(f'Generated {len(manifest["plots"])} charts in {Path(args.run) / "figures"}')


if __name__ == "__main__":
    main()
