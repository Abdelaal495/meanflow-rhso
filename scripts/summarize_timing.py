#!/usr/bin/env python
"""Summarise the rebuttal timing run (Task 1): runtime, peak memory, cost per forward pass.

    python scripts/summarize_timing.py outputs/timing_pmf outputs/timing_jit \
        outputs/timing_imf outputs/timing_sit [--out outputs/timing_summary]

Each argument is a run directory written by scripts/run_timing.py (one per prior).  The
per-job `metadata.json` files are read directly -- they are what `run.py --aggregate` merges
into results.csv, and they also hold the `timing_detail` block (src/timing.py) that the CSV
does not carry.

Per job (one prior, one task, one method):
  * Eval. and BP per image (model_evaluations, backprops_through_model / num_images);
  * seconds per image: mean, SD, min and max over the images, from the OUTER device-
    synchronised per-image timer when it was recorded, else the job mean (`seconds_basis`
    says which); the method's own timer is kept beside it and any disagreement is flagged;
  * PRIMARY memory: the allocator peak.  PyTorch: max_memory_allocated over the timed
    region, exact.  JAX: exact when peak_bytes_in_use rose during the timed region;
    otherwise a BRACKET lower--upper, lower = the largest bytes_in_use sampled during the
    timed region (in the NVML sampling loop), upper = peak_bytes_in_use of the
    one-configuration process; flagged UNINFORMATIVE when the upper bound equals the
    model-only baseline's load peak (then it was set by loading the model);
  * the NVML process figure, which is the process's RESERVED POOL plus CUDA context;
  * both also above the prior's model-only baseline (memory_baseline_<prior>.json);
  * seconds per forward-equivalent = seconds per image / (evaluations + 2 x backprops).

Per prior: the spread of seconds per forward-equivalent across all rows, to show whether
forward-equivalents track wall-clock, and the GPU each prior ran on (a warning is printed
when the priors ran on different GPU types).

Outputs in --out: timing_summary.csv (one row per job), timing_fe_spread.csv (one row per
prior) and timing_tables.tex with three tables in the layout of Table 29 (prior as a
multirow first column, one row per method, one column per task): seconds per image,
allocator peak memory, and cost per forward-equivalent.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

PRIOR_NAME = {"pmf": "pMF-L/16", "jit": "JiT-B/16", "imf": "iMF-B/2", "sit": "SiT-XL/2"}
PRIOR_ORDER = ["pmf", "jit", "imf", "sit"]
TASK_NAME = {"denoising": "Denoising", "deblur": "Deblurring", "super_resolution": "SR $2\\times$",
             "random_inpaint": "Random inpainting", "box_inpaint": "Box inpainting"}
TASK_ORDER = list(TASK_NAME)
# (method, variant) -> label, in table order.  `matched` marks the matched-budget entries.
ROWS = [
    (("sdedit", "base"), "SDEdit"),
    (("mpc_rhc", "base"), "MPC-RHC"),
    (("mpc_rhc", "matched_time"), "MPC-RHC (matched time)"),
    (("mpc_delta_t", "base"), "MPC-$\\Delta t$"),
    (("mpc_delta_t", "matched"), "MPC-$\\Delta t$ (matched BP)"),
    (("pnp", "base"), "PnP-Flow"),
    (("pnp", "matched_time"), "PnP-Flow (matched time)"),
    (("dflow", "base"), "D-Flow"),
    (("dflow", "matched_time"), "D-Flow (matched time)"),
    (("flower", "base"), "Flower"),
    (("flower", "matched_time"), "Flower (matched time)"),
    (("rhso", "base"), "RHSO"),
    (("rhso", "matched"), "RHSO (matched BP)"),
]
ROW_LABEL = dict(ROWS)
ROW_ORDER = {key: i for i, (key, _) in enumerate(ROWS)}
DISAGREEMENT = 0.05            # relative outer-vs-method gap that is flagged ...
DISAGREEMENT_ABS = 0.02        # ... when it also exceeds this many seconds per image
UNINFORMATIVE_BYTES = 2 ** 20  # JAX upper bound within 1 MiB of the model-only load peak

CSV_COLUMNS = [
    "prior", "prior_name", "task", "method", "variant", "row_label", "experiment", "job_id",
    "status", "num_images", "batch_size", "evals_per_image", "bp_per_image",
    "forward_equivalents_per_image", "seconds_per_image_mean", "seconds_per_image_sd",
    "seconds_per_image_min", "seconds_per_image_max", "seconds_per_image_values",
    "seconds_basis", "method_seconds_per_image_mean", "outer_vs_method_rel_diff",
    "seconds_per_forward_equivalent", "allocator_peak_gib", "allocator_peak_exact",
    "allocator_peak_lower_gib", "allocator_peak_upper_gib", "allocator_peak_uninformative",
    "allocator_peak_scope", "allocator_peak_source", "allocator_incremental_over_model_gib",
    "nvml_process_reserved_peak_gib", "nvml_incremental_over_model_gib",
    "model_only_allocator_gib", "model_only_nvml_reserved_gib", "gpu_name",
    "gpu_memory_total_mib", "sync_method", "jax_pool_bytes_before", "jax_pool_bytes_after",
    "jax_bytes_limit", "run_dir",
]
SPREAD_COLUMNS = ["prior", "prior_name", "gpu_names", "rows", "fe_seconds_min",
                  "fe_seconds_max", "fe_max_over_min", "fe_cv", "fe_min_row", "fe_max_row"]


def fnum(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def variant_of(experiment):
    if "__matched_time" in str(experiment):       # Task 5 matched-time entries
        return "matched_time"
    return "matched" if "_matched" in str(experiment) else "base"


def short_gpu(name):
    name = str(name or "").strip()
    return name[len("NVIDIA "):] if name.startswith("NVIDIA ") else name


def load_run(run_dir):
    """(job payloads, {model: model-only baseline})."""
    run_dir = Path(run_dir)
    jobs = {}
    for meta in sorted(run_dir.rglob("metadata.json")):
        try:
            payload = json.loads(meta.read_text())
        except Exception:
            continue
        job_id = payload.get("job_id")
        if job_id and job_id not in jobs:
            payload["_run_dir"] = str(run_dir)
            jobs[job_id] = payload
    baselines = {}
    for path in sorted(run_dir.glob("memory_baseline_*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        baselines[payload.get("model") or path.stem[len("memory_baseline_"):]] = payload
    return list(jobs.values()), baselines


def per_image_seconds(payload):
    """(values, basis, method-timer mean) for one job."""
    detail = payload.get("timing_detail") or {}
    outer = detail.get("chunk_seconds_outer_synced") or []
    sizes = detail.get("chunk_sizes") or []
    if outer and len(sizes) == len(outer) and all(int(s) >= 1 for s in sizes):
        values = []
        for seconds, size in zip(outer, sizes):
            values.extend([float(seconds) / int(size)] * int(size))
        method = detail.get("chunk_seconds_method") or []
        method_mean = (sum(method) / sum(int(s) for s in sizes)
                       if method and len(method) == len(sizes) else None)
        basis = ("per_image_outer_synced" if all(int(s) == 1 for s in sizes)
                 else "per_chunk_outer_synced_divided_by_batch")
        return values, basis, method_mean
    job = fnum(payload.get("runtime_per_image"))
    return ([job] if job is not None else []), "job_mean", job


def summarise_job(payload, baselines):
    detail = payload.get("timing_detail") or {}
    env = detail.get("environment") or {}
    hardware = env.get("hardware") or {}
    model = payload.get("model")
    n = max(1, int(payload.get("num_images") or 1))
    # The counters count BATCHED network calls, so per image means per batch chunk.
    chunks = max(1, -(-n // max(1, int(payload.get("batch_size") or 1))))
    evals = (fnum(payload.get("model_evaluations")) or 0.0) / chunks
    bps = (fnum(payload.get("backprops_through_model")) or 0.0) / chunks
    fe = evals + 2.0 * bps
    values, basis, method_mean = per_image_seconds(payload)
    mean = statistics.fmean(values) if values else None
    sd = statistics.stdev(values) if len(values) > 1 else None
    rel = (abs(mean - method_mean) / mean
           if mean and method_mean is not None and basis != "job_mean" else None)

    alloc = fnum(detail.get("allocator_peak_gib"))
    scope = detail.get("allocator_peak_scope")
    source = detail.get("allocator_peak_source")
    if alloc is None and payload.get("gpu_memory_source") == "torch.cuda.max_memory_allocated":
        alloc, scope, source = (fnum(payload.get("gpu_peak_gib")), "timed_region",
                                "torch.cuda.max_memory_allocated")
    nvml = fnum(payload.get("gpu_process_peak_gib"))
    base = baselines.get(model) or {}
    base_alloc = fnum(base.get("allocator_in_use_gib"))
    base_nvml = fnum(base.get("nvml_process_reserved_gib"))
    exact = detail.get("allocator_peak_exact")
    if exact is None:                            # PyTorch fallback above, or older records
        exact = scope == "timed_region"
    lower = fnum(detail.get("allocator_peak_lower_gib"))
    upper = fnum(detail.get("allocator_peak_upper_gib"))
    if upper is None and alloc is not None:
        upper = alloc
        lower = alloc if exact else lower
    uninformative = False
    if not exact and upper is not None:
        up_bytes = detail.get("allocator_peak_upper_bytes")
        base_bytes = base.get("allocator_peak_during_load_bytes")
        if up_bytes is not None and base_bytes is not None:
            uninformative = abs(int(up_bytes) - int(base_bytes)) <= UNINFORMATIVE_BYTES
        else:
            base_peak = fnum(base.get("allocator_peak_during_load_gib"))
            uninformative = base_peak is not None and abs(upper - base_peak) < 1e-3
    return {
        "prior": model, "prior_name": PRIOR_NAME.get(model, model),
        "task": payload.get("problem"), "method": payload.get("method"),
        "variant": variant_of(payload.get("experiment")),
        "row_label": ROW_LABEL.get((payload.get("method"), variant_of(payload.get("experiment"))),
                                   payload.get("method")),
        "experiment": payload.get("experiment"), "job_id": payload.get("job_id"),
        "status": payload.get("status"), "num_images": payload.get("num_images"),
        "batch_size": payload.get("batch_size"),
        "evals_per_image": evals, "bp_per_image": bps, "forward_equivalents_per_image": fe,
        "seconds_per_image_mean": mean, "seconds_per_image_sd": sd,
        "seconds_per_image_min": min(values) if values else None,
        "seconds_per_image_max": max(values) if values else None,
        "seconds_per_image_values": ";".join("%.6f" % v for v in values),
        "seconds_basis": basis, "method_seconds_per_image_mean": method_mean,
        "outer_vs_method_rel_diff": rel,
        "seconds_per_forward_equivalent": (mean / fe if mean is not None and fe > 0 else None),
        "allocator_peak_gib": alloc, "allocator_peak_scope": scope,
        "allocator_peak_source": source, "allocator_peak_exact": bool(exact),
        "allocator_peak_lower_gib": lower, "allocator_peak_upper_gib": upper,
        "allocator_peak_uninformative": uninformative,
        "allocator_incremental_over_model_gib": (alloc - base_alloc
                                                 if alloc is not None and base_alloc is not None
                                                 else None),
        "nvml_process_reserved_peak_gib": nvml,
        "nvml_incremental_over_model_gib": (nvml - base_nvml
                                            if nvml is not None and base_nvml is not None
                                            else None),
        "model_only_allocator_gib": base_alloc, "model_only_nvml_reserved_gib": base_nvml,
        "gpu_name": hardware.get("gpu_name"),
        "gpu_memory_total_mib": hardware.get("gpu_memory_total_mib"),
        "sync_method": detail.get("sync_method"),
        "jax_pool_bytes_before": detail.get("jax_pool_bytes_before_timed_region"),
        "jax_pool_bytes_after": detail.get("jax_pool_bytes_after_timed_region"),
        "jax_bytes_limit": detail.get("jax_bytes_limit_after_timed_region"),
        "run_dir": payload.get("_run_dir"),
    }


def sort_key(row):
    return (PRIOR_ORDER.index(row["prior"]) if row["prior"] in PRIOR_ORDER else 99,
            ROW_ORDER.get((row["method"], row["variant"]), 99),
            TASK_ORDER.index(row["task"]) if row["task"] in TASK_ORDER else 99)


def collect(run_dirs):
    rows, baselines_all = [], {}
    for run_dir in run_dirs:
        jobs, baselines = load_run(run_dir)
        baselines_all.update(baselines)
        rows.extend(summarise_job(job, baselines) for job in jobs)
    ok = sorted((r for r in rows if r["status"] == "ok"), key=sort_key)
    bad = sorted((r for r in rows if r["status"] != "ok"), key=sort_key)
    return ok, bad, baselines_all


def fe_spread(rows):
    out = []
    for prior in sorted({r["prior"] for r in rows}, key=lambda p: PRIOR_ORDER.index(p)
                        if p in PRIOR_ORDER else 99):
        mine = [r for r in rows if r["prior"] == prior
                and r["seconds_per_forward_equivalent"] is not None]
        gpus = sorted({str(r["gpu_name"]) for r in rows if r["prior"] == prior})
        if not mine:
            continue
        values = [r["seconds_per_forward_equivalent"] for r in mine]
        lo = min(mine, key=lambda r: r["seconds_per_forward_equivalent"])
        hi = max(mine, key=lambda r: r["seconds_per_forward_equivalent"])
        mean = statistics.fmean(values)
        out.append({
            "prior": prior, "prior_name": PRIOR_NAME.get(prior, prior),
            "gpu_names": ";".join(gpus), "rows": len(mine),
            "fe_seconds_min": min(values), "fe_seconds_max": max(values),
            "fe_max_over_min": (max(values) / min(values)) if min(values) > 0 else None,
            "fe_cv": (statistics.stdev(values) / mean) if len(values) > 1 and mean else None,
            "fe_min_row": "%s / %s" % (lo["row_label"], lo["task"]),
            "fe_max_row": "%s / %s" % (hi["row_label"], hi["task"])})
    return out


def gpu_warnings(rows):
    """Warnings when a prior ran on several GPU types, or the priors on different ones."""
    warnings, per_prior = [], {}
    for r in rows:
        per_prior.setdefault(r["prior"], set()).add(r["gpu_name"])
    for prior, names in sorted(per_prior.items()):
        if len(names) > 1:
            warnings.append("WARNING: %s ran on more than one GPU type: %s"
                            % (prior, sorted(str(n) for n in names)))
        if None in names:
            warnings.append("WARNING: %s has job(s) with no recorded GPU name" % prior)
    known = {n for names in per_prior.values() for n in names if n}
    if len(known) > 1:
        warnings.append("WARNING: the priors ran on DIFFERENT GPU types (%s); their runtimes "
                        "and memory are not comparable in one table."
                        % ", ".join(sorted(known)))
    return warnings, per_prior


def caption_gpu(rows):
    known = {short_gpu(r["gpu_name"]) for r in rows if r["gpu_name"]}
    return known.pop() if len(known) == 1 else ("mixed GPUs" if known else "GPU unknown")


# ------------------------------------------------------------------ LaTeX
def _cell(value, fmt, mark=""):
    return "--" if value is None else (fmt % value) + mark


def memory_cell(r, latex=True):
    """Exact value (PyTorch; JAX when its peak rose in the timed region) or lower--upper."""
    if r["allocator_peak_exact"]:
        return _cell(r["allocator_peak_gib"], "%.2f")
    lo, hi = r["allocator_peak_lower_gib"], r["allocator_peak_upper_gib"]
    if hi is None:
        return "--"
    mark = ("$^\\ast$" if latex else "*") if r["allocator_peak_uninformative"] else ""
    return "%s%s%.2f%s" % ("?" if lo is None else "%.2f" % lo, "--" if latex else "-",
                           hi, mark)


def latex_table(rows, baselines, quantity, gpu, fe_unit="ms"):
    """One table in Table 29's layout: multirow prior, one row per method, one column per task."""
    if quantity == "seconds":
        caption = "Runtime per image (seconds/image, batch size one, %s)" % gpu
        label = "tab:rebuttal-runtime"
    elif quantity == "memory":
        caption = ("Peak GPU memory per job (allocator high-water mark, GiB, batch size one, "
                   "%s; JAX: exact, or lower--upper bound)" % gpu)
        label = "tab:rebuttal-memory"
    else:
        unit = "milliseconds" if fe_unit == "ms" else "seconds"
        caption = ("Cost per forward-equivalent (%s per image / (evaluations + 2 "
                   "$\\times$ backprops), batch size one, %s)" % (unit, gpu))
        label = "tab:rebuttal-forward-equivalent"
    lines = ["% Generated by scripts/summarize_timing.py; needs \\usepackage{booktabs,multirow}.",
             "\\begin{table}[t]", "\\centering", "\\caption{%s}" % caption,
             "\\label{%s}" % label, "\\begin{tabular}{ll%s}" % ("c" * len(TASK_ORDER)),
             "\\toprule",
             "Prior & Method & %s \\\\" % " & ".join(TASK_NAME[t] for t in TASK_ORDER),
             "\\midrule"]
    priors = [p for p in PRIOR_ORDER if any(r["prior"] == p for r in rows)]
    dagger = False
    for pi, prior in enumerate(priors):
        mine = {(r["method"], r["variant"], r["task"]): r for r in rows if r["prior"] == prior}
        keys = [key for key, _ in ROWS if any((key[0], key[1], t) in mine for t in TASK_ORDER)]
        body = []
        for key in keys:
            cells = []
            for task in TASK_ORDER:
                r = mine.get((key[0], key[1], task))
                if r is None:
                    cells.append("--")
                elif quantity == "seconds":
                    cells.append(_cell(r["seconds_per_image_mean"], "%.2f"))
                elif quantity == "memory":
                    dagger = dagger or not r["allocator_peak_exact"]
                    cells.append(memory_cell(r))
                else:
                    v = r["seconds_per_forward_equivalent"]
                    v = None if v is None else (v * 1000.0 if fe_unit == "ms" else v)
                    cells.append(_cell(v, "%.2f" if fe_unit == "ms" else "%.4f"))
            body.append((ROW_LABEL[key], cells))
        if quantity == "memory":
            base = fnum((baselines.get(prior) or {}).get("allocator_in_use_gib"))
            body.append(("Model only", None if base is None else "%.2f" % base))
        n = len(body)
        for i, (label_, cells) in enumerate(body):
            first = ("\\multirow{%d}{*}{%s}" % (n, PRIOR_NAME.get(prior, prior))) if i == 0 else ""
            if isinstance(cells, list):
                lines.append("%s & %s & %s \\\\" % (first, label_, " & ".join(cells)))
            else:
                lines.append("%s & %s & \\multicolumn{%d}{c}{%s} \\\\"
                             % (first, label_, len(TASK_ORDER), cells or "--"))
        lines.append("\\midrule" if pi < len(priors) - 1 else "\\bottomrule")
    if not priors:
        lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table}"]
    if dagger:
        lines.insert(-2, "% a--b: JAX bracket. a = largest bytes_in_use sampled during the "
                         "timed region (lower bound); b = peak_bytes_in_use of the "
                         "one-configuration process, set before the timed region (upper "
                         "bound). $^\\ast$: b equals the model-only load peak, so it says "
                         "nothing about the method.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ text output
def print_tables(rows, baselines, per_prior_gpus):
    for prior in [p for p in PRIOR_ORDER if any(r["prior"] == p for r in rows)]:
        gpus = ", ".join(sorted(str(n) for n in per_prior_gpus.get(prior, {None})))
        print("\n%s  --  GPU: %s" % (PRIOR_NAME.get(prior, prior), gpus))
        base = baselines.get(prior) or {}
        print("  model-only baseline: allocator %s GiB in use, NVML reserved %s GiB"
              % (_cell(fnum(base.get("allocator_in_use_gib")), "%.3f"),
                 _cell(fnum(base.get("nvml_process_reserved_gib")), "%.3f")))
        print("  %-28s %-17s %8s %7s %16s %10s %11s %9s %9s"
              % ("method", "task", "Eval.", "BP", "s/img mean+-sd", "ms/FE", "alloc GiB",
                 "+model", "NVML GiB"))
        for r in (r for r in rows if r["prior"] == prior):
            sd = r["seconds_per_image_sd"]
            timing = ("%.3f+-%.3f" % (r["seconds_per_image_mean"], sd) if sd is not None
                      else _cell(r["seconds_per_image_mean"], "%.3f") + " (job)")
            fe = r["seconds_per_forward_equivalent"]
            print("  %-28s %-17s %8.0f %7.0f %16s %10s %11s %9s %9s%s"
                  % (r["row_label"].replace("$\\Delta t$", "dt"), r["task"], r["evals_per_image"],
                     r["bp_per_image"], timing, _cell(None if fe is None else fe * 1000, "%.3f"),
                     memory_cell(r, latex=False),
                     _cell(r["allocator_incremental_over_model_gib"], "%.3f"),
                     _cell(r["nvml_process_reserved_peak_gib"], "%.3f"),
                     "  UNINFORMATIVE" if r["allocator_peak_uninformative"] else ""))
    print("\n  alloc GiB: exact, or lo-hi for JAX (lo = max bytes_in_use sampled in the timed "
          "region, hi = lifetime peak_bytes_in_use of the one-configuration process).")
    print("  UNINFORMATIVE: hi equals the model-only baseline's load peak, so it was set by "
          "loading the model and says nothing about the method.")
    print("  NVML GiB is the process's RESERVED pool plus CUDA context, not live tensors.")


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in columns})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dirs", nargs="+", help="run directories written by run_timing.py")
    p.add_argument("--out", default="outputs/timing_summary")
    p.add_argument("--fe-unit", choices=("ms", "s"), default="ms",
                   help="unit of the forward-equivalent table (default ms: seconds would "
                        "print 0.00 at two decimals)")
    args = p.parse_args(argv)

    rows, bad, baselines = collect(args.run_dirs)
    if not rows:
        print("No finished job found under %s" % ", ".join(args.run_dirs))
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spread = fe_spread(rows)
    write_csv(out / "timing_summary.csv", rows, CSV_COLUMNS)
    write_csv(out / "timing_fe_spread.csv", spread, SPREAD_COLUMNS)
    gpu = caption_gpu(rows)
    tex = "\n".join(latex_table(rows, baselines, q, gpu, args.fe_unit)
                    for q in ("seconds", "memory", "forward_equivalent"))
    (out / "timing_tables.tex").write_text(tex)

    warnings, per_prior = gpu_warnings(rows)
    print_tables(rows, baselines, per_prior)
    print("\nSeconds per forward-equivalent, spread across methods and tasks:")
    for s in spread:
        print("  %-9s %3d rows  min %.2f ms (%s)  max %.2f ms (%s)  max/min %.2f  CV %s"
              % (s["prior_name"], s["rows"], s["fe_seconds_min"] * 1000, s["fe_min_row"],
                 s["fe_seconds_max"] * 1000, s["fe_max_row"], s["fe_max_over_min"] or float("nan"),
                 _cell(s["fe_cv"], "%.2f")))
    flagged = [r for r in rows if r["outer_vs_method_rel_diff"] is not None
               and r["outer_vs_method_rel_diff"] > DISAGREEMENT
               and abs(r["seconds_per_image_mean"] - r["method_seconds_per_image_mean"])
               > DISAGREEMENT_ABS]
    for r in flagged:
        print("NOTE: %s %s %s: outer synchronised %.3f s/img vs method timer %.3f s/img"
              % (r["prior"], r["row_label"], r["task"], r["seconds_per_image_mean"],
                 r["method_seconds_per_image_mean"]))
    for r in bad:
        print("NOT OK: %s %s %s status=%s" % (r["prior"], r["row_label"], r["task"], r["status"]))
    missing_base = sorted({r["prior"] for r in rows} - set(baselines))
    if missing_base:
        print("WARNING: no model-only baseline for %s" % ", ".join(missing_base))
    for w in warnings:
        print(w)
    print("\nwrote %s, %s, %s" % (out / "timing_summary.csv", out / "timing_fe_spread.csv",
                                  out / "timing_tables.tex"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
