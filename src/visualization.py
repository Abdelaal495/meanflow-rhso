"""Figures: per-task comparison grids and aggregate summary plots.

Two rules from the brief drive the design:

  * every requested reconstruction is shown -- if a task has more configurations than fit in
    one readable figure, the grid PAGINATES rather than silently dropping the tail;
  * every panel is labelled with the model, the method, t0, K, the MPC step count and
    lambda, so a figure can never be misattributed.  `beta` joins that list whenever it
    differs from the uniform default, so beta-distinct configurations can never be merged
    into one bar or one legend entry.

Nothing here is a dashboard; they are matplotlib figures written to disk and, in a notebook,
displayed inline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import COMPARED_METHODS
from .utils import in_ipython, to_uint8

MAX_COLUMNS_PER_PAGE = 6          # reconstruction columns, on top of the 2 reference columns
MAX_ROWS = 4                      # images per page


def _setup():
    import matplotlib
    if not in_ipython():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["figure.max_open_warning"] = 0
    return plt


def _finish(fig, path: Path, show: bool) -> Path:
    plt = _setup()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    if show and in_ipython():
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_source_images(images: np.ndarray, names: Sequence[str], labels: Sequence[int],
                       path: Path, show: bool = True, limit: int = 8) -> Path:
    plt = _setup()
    k = min(limit, len(images))
    fig, axes = plt.subplots(1, k, figsize=(2.0 * k, 2.5), squeeze=False)
    for i in range(k):
        axes[0][i].imshow(to_uint8(images[i]))
        axes[0][i].axis("off")
        axes[0][i].set_title("%s\n(class %d)" % (names[i], labels[i]), fontsize=8)
    fig.suptitle("Ground truth -- one shared pool for every experiment, model and method",
                 fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_problem_overview(problems: Dict[str, Any], path: Path, show: bool = True,
                          n_show: int = 3) -> Path:
    """One row per problem: ground truth, degraded observation, initialisation guide."""
    plt = _setup()
    rows = len(problems)
    # Fit the grid to the images that actually exist: allocating 3 columns per slot and then
    # filling only as many as the run has left empty axes on the right.
    available = min(len(p.measurement) for p in problems.values())
    n_show = max(1, min(int(n_show), int(available)))
    fig, axes = plt.subplots(rows, 3 * n_show, figsize=(2.0 * 3 * n_show, 2.4 * rows),
                             squeeze=False)
    for r, (_key, p) in enumerate(problems.items()):
        for i in range(n_show):
            panels = [("ground truth", p.ground_truth[i]),
                      ("degraded y", p.display_measurement[i]),
                      ("guide g(y)", p.initialization_guide[i])]
            for c, (title, img) in enumerate(panels):
                ax = axes[r][3 * i + c]
                ax.imshow(to_uint8(img))
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if r == 0:
                    ax.set_title(title, fontsize=8)
                if 3 * i + c == 0:
                    ax.set_ylabel(p.name, fontsize=8)
    fig.suptitle("Inverse-problem instances: y = A(x*) + eta, built once and shared by "
                 "SDEdit, MPC-RHC and MPC-delta_t", fontweight="bold")
    fig.tight_layout()
    return _finish(fig, path, show)


def plot_comparison_grid(experiment: str, model: str, records: Sequence[Dict[str, Any]],
                         problem, figures_dir: Path, show: bool = True,
                         max_rows: int = MAX_ROWS,
                         max_columns: int = MAX_COLUMNS_PER_PAGE) -> List[Path]:
    """Ground truth | degraded | every requested reconstruction, paginated.

    `records` are dicts with keys 'title' and 'images' (an (N,256,256,3) array).  Sorted so
    that SDEdit appears first, which makes the MPC columns read as deltas against it.
    """
    plt = _setup()
    if not records:
        return []
    order = {"sdedit": 0, "mpc_rhc": 1, "mpc_delta_t": 2, "pnp": 3, "dflow": 4, "rhso": 5,
             "flower": 6}
    records = sorted(records, key=lambda r: (order.get(r.get("method", ""), 9),
                                             r.get("t0", 0.0), r.get("sort_key", "")))
    n_images = min(max_rows, len(problem.ground_truth))
    pages = [records[i:i + max_columns] for i in range(0, len(records), max_columns)]
    paths: List[Path] = []

    for page_idx, page in enumerate(pages, 1):
        n_cols = 2 + len(page)
        fig, axes = plt.subplots(n_images, n_cols,
                                 figsize=(2.05 * n_cols, 2.35 * n_images), squeeze=False)
        for row in range(n_images):
            panels = [("original", problem.ground_truth[row]),
                      ("degraded /\nguide", problem.display_measurement[row])]
            panels += [(rec["title"], rec["images"][row]) for rec in page]
            for col, (title, img) in enumerate(panels):
                ax = axes[row][col]
                ax.imshow(to_uint8(img))
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                if row == 0:
                    ax.set_title(title, fontsize=7.5, linespacing=1.25)
            axes[row][0].set_ylabel(problem.image_ids[row], fontsize=6.5)
        suffix = ("  (page %d/%d)" % (page_idx, len(pages))) if len(pages) > 1 else ""
        fig.suptitle("%s | %s  --  same image, same y, same t0, same epsilon%s"
                     % (experiment, model.upper(), suffix), fontweight="bold", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        paths.append(_finish(fig, figures_dir / ("%s_%s_page_%02d.png"
                                                 % (experiment, model, page_idx)), show))
    return paths


CONFIG_COLORS = {"sdedit": "#4C72B0", "mpc_rhc": "#DD8452", "mpc_delta_t": "#55A868",
                 "pnp": "#8172B3", "dflow": "#C44E52", "rhso": "#937860",
                 "flower": "#DA8BC3"}


def _beta_suffix(row: Dict[str, Any]) -> List[str]:
    """`b=<beta>`, but only when beta is not the uniform default.

    Two configurations that differ only in beta are DIFFERENT configurations and must never
    share a bar, a colour slot or a legend entry; a run that leaves beta at 1 everywhere
    gets no extra label clutter.
    """
    beta = row.get("beta")
    if beta is None or float(beta) == 1.0:
        return []
    return ["b=%g" % float(beta)]


def config_label(row: Dict[str, Any]) -> str:
    """A short label identifying ONE resolved configuration -- never a group of them."""
    method = row["method"]
    if method == "sdedit":
        parts = ["SDEdit", "n=%s" % row.get("steps")]
        if row.get("solver"):
            parts.append(str(row["solver"]))
    elif method == "mpc_rhc":
        parts = ["RHC", "K=%s" % row.get("K"), "N=%s" % row.get("num_mpc_steps"),
                 "lam=%g" % (row.get("lam") or 0)]
    elif method == "pnp":
        parts = ["PnP", "N=%s" % row.get("num_pnp_steps"),
                 "g0=%g" % (row.get("gamma0") or 0), "a=%g" % (row.get("alpha") or 0)]
        if (row.get("noise_samples") or 1) != 1:
            parts.append("M=%s" % row.get("noise_samples"))
    elif method == "dflow":
        parts = ["D-Flow", "n=%s" % row.get("steps"), "opt=%s" % row.get("num_opt_steps"),
                 "lr=%g" % (row.get("lr") or 0)]
    elif method == "rhso":
        parts = ["RHSO", "N=%s" % row.get("num_rhso_steps"),
                 "opt=%s" % row.get("num_opt_steps"), "lr=%g" % (row.get("lr") or 0)]
    elif method == "flower":
        parts = ["Flower", "N=%s" % row.get("num_flower_steps")]
        if str(row.get("flower_gamma") or 0) not in ("0", "0.0"):
            parts.append("gamma=%s" % row.get("flower_gamma"))
    else:
        parts = ["MPC-dt", "N=%s" % row.get("num_mpc_steps"), "lam=%g" % (row.get("lam") or 0)]
    return " ".join(parts + _beta_suffix(row))


def group_label(row: Dict[str, Any]) -> str:
    """Configuration identity WITHOUT lambda, for grouping bars across tasks.

    Table E2 gives a different lambda per task, so including it would make every task its
    own bar group and leave the grid full of gaps.  Lambda is not hidden: it is annotated
    on each bar, and the tables and the paired-delta plot use the full label.
    """
    method = row["method"]
    suffix = "".join(" " + p for p in _beta_suffix(row))
    if method == "sdedit":
        parts = ["SDEdit", "n=%s" % row.get("steps")]
        if row.get("solver"):
            parts.append(str(row["solver"]))
        return " ".join(parts) + suffix
    if method == "mpc_rhc":
        return "RHC K=%s N=%s%s" % (row.get("K"), row.get("num_mpc_steps"), suffix)
    if method == "pnp":
        # gamma0/alpha play the role lambda plays for MPC (they scale the data term), so
        # they are dropped here for the same reason and annotated on the bar instead.
        return "PnP N=%s%s%s" % (row.get("num_pnp_steps"),
                                 "" if (row.get("noise_samples") or 1) == 1
                                 else " M=%s" % row.get("noise_samples"), suffix)
    if method == "dflow":
        return "D-Flow n=%s opt=%s%s" % (row.get("steps"), row.get("num_opt_steps"), suffix)
    if method == "rhso":
        # beta is NOT dropped here: it changes the executed trajectory, unlike lambda, which
        # only reweights an objective.
        return "RHSO N=%s opt=%s%s" % (row.get("num_rhso_steps"),
                                       row.get("num_opt_steps"), suffix)
    if method == "flower":
        gamma = str(row.get("flower_gamma") or 0)
        return "Flower N=%s%s%s" % (row.get("num_flower_steps"),
                                    "" if gamma in ("0", "0.0") else " gamma=%s" % gamma,
                                    suffix)
    return "MPC-dt N=%s%s" % (row.get("num_mpc_steps"), suffix)


def _method_rank(method: str) -> int:
    return {"sdedit": 0, "mpc_rhc": 1, "mpc_delta_t": 2, "pnp": 3, "dflow": 4,
            "rhso": 5, "flower": 6}.get(method, 9)


def _shade(base_hex: str, index: int, total: int):
    """Distinguish configurations WITHIN a method without changing its identity colour."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(base_hex)
    if total <= 1:
        return (r, g, b)
    f = 0.55 + 0.45 * (index / max(1, total - 1))          # darker -> lighter
    return tuple(min(1.0, c * f + (1.0 - f) * 0.15 + 0.0) for c in (r, g, b))


def plot_configuration_breakdown(rows: Sequence[Dict[str, Any]], model: str, path: Path,
                                 show: bool = True, max_configs: int = 12) -> Optional[Path]:
    """One bar per RESOLVED CONFIGURATION -- nothing is averaged across models or settings.

    Averaging over configurations would mix a 4-step SDEdit with a 25-step one, and a
    lambda=0.04 RHC with a lambda=1 one, producing a number that describes no run that was
    actually executed.  Each bar here is one atomic job.  The dashed line is the degraded
    observation itself (the "do nothing" baseline), which is the reference that decides
    whether a reconstruction helped at all.
    """
    plt = _setup()
    rows = [r for r in rows if r.get("status") == "ok" and r.get("model") == model]
    if not rows:
        return None
    tasks = sorted({r["task"] for r in rows})

    seen, configs = set(), []
    for r in sorted(rows, key=lambda r: (_method_rank(r["method"]), group_label(r))):
        label = group_label(r)
        if label not in seen:
            seen.add(label)
            configs.append((label, r["method"]))
    truncated = len(configs) > max_configs
    configs = configs[:max_configs]

    per_method_index, per_method_total = {}, {}
    for label, method in configs:
        per_method_total[method] = per_method_total.get(method, 0) + 1
    counter = {}
    for label, method in configs:
        per_method_index[label] = counter.get(method, 0)
        counter[method] = counter.get(method, 0) + 1

    metrics = [("PSNR (dB)", "psnr", "degraded_psnr"), ("SSIM", "ssim", "degraded_ssim"),
               ("LPIPS (lower is better)", "lpips", "degraded_lpips"),
               ("runtime / image (s)", "runtime_per_image", None)]
    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(max(9.0, 1.9 * len(tasks) * 1.6), 3.3 * len(metrics)),
                             squeeze=False)
    width = 0.84 / max(1, len(configs))
    base = np.arange(len(tasks), dtype=float)

    for m_idx, (title, key, degraded_key) in enumerate(metrics):
        ax = axes[m_idx][0]
        drew_any = False
        for c_idx, (label, method) in enumerate(configs):
            values, lams = [], []
            for task in tasks:
                match = [r for r in rows
                         if r["task"] == task and group_label(r) == label
                         and r.get(key) is not None]
                values.append(float(match[0][key]) if match else np.nan)
                lams.append(match[0].get("lam") if match else None)
            drew_any = drew_any or any(np.isfinite(v) for v in values)
            bars = ax.bar(base + c_idx * width, values, width,
                          color=_shade(CONFIG_COLORS.get(method, "grey"),
                                       per_method_index[label], per_method_total[method]),
                          label=label if m_idx == 0 else None, edgecolor="white",
                          linewidth=0.4)
            if method != "sdedit":
                # Lambda is task-dependent (Table E2), so it is shown per bar rather than
                # folded into the group label.
                for bar, lam in zip(bars, lams):
                    if lam is not None and np.isfinite(bar.get_height()):
                        ax.annotate("λ=%g" % lam,
                                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    ha="center", va="bottom", fontsize=5.5, rotation=90,
                                    xytext=(0, 2), textcoords="offset points", alpha=0.8)
        if degraded_key:
            for t_idx, task in enumerate(tasks):
                vals = [r[degraded_key] for r in rows
                        if r["task"] == task and r.get(degraded_key) is not None]
                if vals:
                    x0 = base[t_idx] - width / 2
                    x1 = base[t_idx] + width * (len(configs) - 0.5)
                    ax.plot([x0, x1], [vals[0], vals[0]], color="black", linestyle="--",
                            linewidth=1.3, zorder=5,
                            label="degraded observation (no reconstruction)"
                            if (m_idx == 0 and t_idx == 0) else None)
        ax.set_xticks(base + width * (len(configs) - 1) / 2)
        ax.set_xticklabels(tasks, rotation=12, ha="right", fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        if key == "runtime_per_image":
            ax.set_yscale("log")
        if not drew_any:
            ax.text(0.5, 0.5, "%s not available in this run" % title.split(" (")[0],
                    transform=ax.transAxes, ha="center", va="center", fontsize=10,
                    color="grey", style="italic")
        else:
            ax.margins(y=0.18)          # headroom for the lambda annotations
        if m_idx == 0:
            # Above the axes, so the degraded-baseline line is never hidden behind it.
            ax.legend(fontsize=7, ncol=min(4, len(configs) + 1), loc="lower left",
                      bbox_to_anchor=(0.0, 1.02), frameon=False)

    suffix = "  (showing the first %d configurations)" % max_configs if truncated else ""
    fig.suptitle("%s -- every resolved configuration shown separately, no averaging%s"
                 % (model.upper(), suffix), fontweight="bold", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    return _finish(fig, path, show)


def step_matched_baseline(row: Dict[str, Any],
                          candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The SDEdit job to compare a non-baseline job against.

    Restricted to the same task, model and t0 -- so the measurement, the guide and epsilon
    are identical -- and then the SDEdit configuration whose trajectory discretisation is
    closest to this job's own. Comparing against an averaged SDEdit would compare against a
    run that never happened; comparing against the best SDEdit at any step count would
    change the baseline between panels.

    "Closest step count" is a WEAK notion of matched cost across methods: a PnP correction
    is one denoiser evaluation, an MPC replan is a planning loop, and a D-Flow step is a
    trajectory that is also differentiated. The baseline is chosen this way so the pairing
    is defined and reproducible, not because the two jobs cost the same -- which is exactly
    why the runtime ratio and the GPU memory columns are reported next to every delta.
    """
    pool = [c for c in candidates
            if c["method"] == "sdedit" and c["task"] == row["task"]
            and c["model"] == row["model"] and c["t0"] == row["t0"]]
    if not pool:
        return None
    target = row.get("num_mpc_steps") or row.get("reconstruction_steps") or 1
    # Among equally step-matched candidates, prefer the SDEdit job on the SAME time
    # schedule: beta does not break the pairing invariant (same problem, model, t0 and
    # epsilon), but comparing like with like is still the better default.
    return min(pool, key=lambda c: (abs((c.get("reconstruction_steps") or 0) - target),
                                    (c.get("beta") != row.get("beta")),
                                    c.get("reconstruction_steps") or 0))


def plot_paired_deltas(rows: Sequence[Dict[str, Any]], path: Path,
                       show: bool = True) -> Optional[Path]:
    """Improvement of each non-baseline configuration over its OWN step-matched SDEdit.

    Every bar is one atomic job minus one atomic job, sharing image, measurement, guide,
    t0, epsilon and conditioning.  Nothing is pooled across models or hyperparameters.
    """
    plt = _setup()
    rows = [r for r in rows if r.get("status") == "ok"]
    mpc = [r for r in rows if r["method"] in COMPARED_METHODS]
    if not mpc:
        return None

    entries = []
    for r in mpc:
        base = step_matched_baseline(r, rows)
        if base is None:
            continue
        entries.append({
            "task": r["task"], "model": r["model"], "method": r["method"],
            "label": "%s / %s" % (r["model"].upper(), group_label(r)),
            "baseline": config_label(base),
            "dpsnr": (r["psnr"] - base["psnr"]) if None not in (r["psnr"], base["psnr"]) else np.nan,
            "dlpips": (r["lpips"] - base["lpips"])
                      if None not in (r.get("lpips"), base.get("lpips")) else np.nan,
            "ratio": ((r["runtime_per_image"] / base["runtime_per_image"])
                      if base.get("runtime_per_image") else np.nan)})
    if not entries:
        return None

    tasks = sorted({e["task"] for e in entries})
    labels = sorted({e["label"] for e in entries},
                    key=lambda s: (_method_rank("mpc_delta_t" if "MPC-dt" in s else "mpc_rhc"), s))
    hatches = {"mpc_rhc": "", "mpc_delta_t": "//"}
    model_colors = {"jit": "#4C72B0", "pmf": "#DD8452", "sit": "#55A868", "imf": "#C44E52"}

    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, 2.4 * len(tasks)), 9), squeeze=False)
    width = 0.82 / max(1, len(labels))
    base_x = np.arange(len(tasks), dtype=float)

    for panel, (key, title, good) in enumerate(
            [("dpsnr", "delta PSNR (dB) vs step-matched SDEdit", "up"),
             ("dlpips", "delta LPIPS vs step-matched SDEdit", "down")]):
        ax = axes[panel][0]
        for l_idx, label in enumerate(labels):
            values, ratios = [], []
            for task in tasks:
                match = [e for e in entries if e["task"] == task and e["label"] == label]
                values.append(match[0][key] if match else np.nan)
                ratios.append(match[0]["ratio"] if match else np.nan)
            model = label.split(" / ")[0].lower()
            method = "mpc_delta_t" if "MPC-dt" in label else "mpc_rhc"
            bars = ax.bar(base_x + l_idx * width, values, width,
                          color=model_colors.get(model, "grey"), alpha=0.9,
                          hatch=hatches[method], edgecolor="white", linewidth=0.5,
                          label=label if panel == 0 else None)
            if panel == 1:
                for bar, ratio in zip(bars, ratios):
                    if np.isfinite(ratio):
                        ax.annotate("%.0fx" % ratio,
                                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    ha="center", fontsize=6, rotation=90,
                                    va="bottom" if bar.get_height() >= 0 else "top",
                                    xytext=(0, 2 if bar.get_height() >= 0 else -2),
                                    textcoords="offset points")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(base_x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(tasks, rotation=12, ha="right", fontsize=8)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.005, 0.97, "improvement is %swards" % good, transform=ax.transAxes,
                fontsize=7.5, va="top", style="italic")
        if panel == 0:
            ax.legend(fontsize=7.5, ncol=min(3, len(labels)))
    axes[1][0].text(0.005, 0.03,
                    "annotations: runtime multiple vs the same baseline | lambda is "
                    "task-dependent (Table E2) and is listed in the results table",
                    transform=axes[1][0].transAxes, fontsize=7, style="italic")

    fig.suptitle("Paired improvement over SDEdit -- one atomic job minus one atomic job\n"
                 "(same image, measurement, guide, t0, epsilon and conditioning)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path, show)


def plot_quality_vs_cost(rows: Sequence[Dict[str, Any]], path: Path,
                         show: bool = True) -> Optional[Path]:
    """LPIPS versus runtime, and measurement consistency versus perceptual quality."""
    plt = _setup()
    rows = [r for r in rows if r.get("status") == "ok"]
    usable = [r for r in rows if r.get("lpips") is not None]
    if not usable:
        usable = rows
    if not usable:
        return None
    markers = {"sdedit": "o", "mpc_rhc": "s", "mpc_delta_t": "^", "pnp": "D",
               "dflow": "P", "rhso": "*", "flower": "v"}
    colors = {"jit": "tab:blue", "pmf": "tab:orange", "sit": "tab:green", "imf": "tab:red"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in usable:
        style = dict(marker=markers.get(r["method"], "x"),
                     color=colors.get(r["model"], "grey"), s=48, alpha=0.85)
        y = r.get("lpips")
        if y is not None and r.get("runtime_per_image"):
            axes[0].scatter(r["runtime_per_image"], y, **style)
        if y is not None and r.get("measurement_rmse") is not None:
            axes[1].scatter(r["measurement_rmse"], y, **style)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("runtime per image (s, log scale)")
    axes[0].set_ylabel("LPIPS (lower is better)")
    axes[0].set_title("Perceptual quality vs compute")
    axes[1].set_xlabel("measurement RMSE  ||A(x_hat) - y||")
    axes[1].set_ylabel("LPIPS (lower is better)")
    axes[1].set_title("Measurement consistency vs perceptual quality")
    for ax in axes:
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker=markers[m], linestyle="", color="black", label=m)
               for m in markers if any(r["method"] == m for r in usable)]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=colors[m], label=m.upper())
                for m in colors if any(r["model"] == m for r in usable)]
    axes[0].legend(handles=handles, fontsize=8)
    fig.suptitle("MPC buys measurement consistency; the question is what it costs and whether "
                 "perceptual quality follows", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _finish(fig, path, show)


def plot_quality_vs_memory(rows: Sequence[Dict[str, Any]], path: Path,
                           show: bool = True) -> Optional[Path]:
    """Perceptual quality against GPU memory, and memory against runtime.

    The memory axis is the INCREMENTAL peak: what a method added on top of an already
    resident model, which is the number that differs between strategies.  Each point is one
    atomic job at its own batch size, and the annotation says so -- a peak measured at
    batch 2 is not two images' worth of anything, and nothing here divides by the batch.
    """
    plt = _setup()
    rows = [r for r in rows if r.get("status") == "ok"
            and r.get("gpu_incremental_peak_gib") is not None]
    if not rows:
        return None
    markers = {"sdedit": "o", "mpc_rhc": "s", "mpc_delta_t": "^", "pnp": "D", "dflow": "P",
               "rhso": "*", "flower": "v"}
    colors = {"jit": "tab:blue", "pmf": "tab:orange", "sit": "tab:green", "imf": "tab:red"}
    batches = sorted({r.get("batch_size") for r in rows if r.get("batch_size")})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for r in rows:
        style = dict(marker=markers.get(r["method"], "x"),
                     color=colors.get(r["model"], "grey"), s=48, alpha=0.85)
        memory = r["gpu_incremental_peak_gib"]
        quality = r.get("lpips") if r.get("lpips") is not None else r.get("psnr")
        if quality is not None:
            axes[0].scatter(memory, quality, **style)
        if r.get("runtime_per_image"):
            axes[1].scatter(r["runtime_per_image"], memory, **style)

    uses_lpips = any(r.get("lpips") is not None for r in rows)
    axes[0].set_xlabel("incremental GPU peak (GiB above the resident model)")
    axes[0].set_ylabel("LPIPS (lower is better)" if uses_lpips else "PSNR (dB)")
    axes[0].set_title("Quality vs the memory the method itself needs")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("runtime per image (s, log scale)")
    axes[1].set_ylabel("incremental GPU peak (GiB)")
    axes[1].set_title("The two costs, against each other")
    for ax in axes:
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([], [], marker=markers[m], linestyle="", color="black", label=m)
               for m in markers if any(r["method"] == m for r in rows)]
    handles += [plt.Line2D([], [], marker="o", linestyle="", color=colors[m], label=m.upper())
                for m in colors if any(r["model"] == m for r in rows)]
    axes[0].legend(handles=handles, fontsize=8)
    sources = sorted({(r.get("gpu_memory_source") or "").split("(")[0] for r in rows})
    fig.suptitle("Memory is a JOB peak at batch %s, never divided by the batch  |  source: %s"
                 % ("/".join(str(b) for b in batches) or "?", ", ".join(s for s in sources)),
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig, path, show)


def plot_stroke_parity(reference: np.ndarray, rendered: np.ndarray, truth: np.ndarray,
                       path: Path, show: bool = True, limit: int = 3) -> Path:
    """Original PIL stroke transform vs the frozen-geometry differentiable renderer."""
    plt = _setup()
    k = min(limit, len(truth))
    fig, axes = plt.subplots(k, 3, figsize=(7.5, 2.5 * k), squeeze=False)
    for i in range(k):
        for c, (title, img) in enumerate((("ground truth", truth[i]),
                                          ("original SDEdit stroke\n(PIL, non-differentiable)",
                                           reference[i]),
                                          ("frozen-geometry renderer\n(differentiable, = y)",
                                           rendered[i]))):
            axes[i][c].imshow(to_uint8(img))
            axes[i][c].axis("off")
            if i == 0:
                axes[i][c].set_title(title, fontsize=8)
    fig.suptitle("Stroke operator parity", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig, path, show)



