"""Hexagon study (rebuttal Task 4): the data and the metrics, with no trained model.

1. the hexagon: regular, side 2, centred, flat lower edge, lower-right corner (1, -sqrt 3);
2. the distance to the hexagon is exact on known points (vertices, edge midpoints, the
   centre, points outside an edge and beyond a corner) and agrees with a dense polyline;
3. the data: uniform in arc length (each edge gets 1/6), centred, and its distance to the
   hexagon follows the jitter (|N(0, 0.02^2 I)| projected on the edge normal, away from
   corners: mean 0.02 sqrt(2 / pi));
4. sliced W2 is right on a known pair of samples;
5. stage 2: the exact posterior of x given y = x_1 + sigma eps (1-D quadrature along the
   hexagon) matches importance sampling of the data model (mean within Monte Carlo error,
   and its samples' mean), for sigma 0.1 and 0.3, with both branches present where the
   vertical line meets the hexagon twice; and the solvers' optional `trace` changes no
   output and records x0 plus one state per outer step.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "toy"))

import math
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

import hexagon_study as H  # noqa: E402

RESULTS = []


def report(name, ok, detail=""):
    print("  [%s] %-50s %s" % ("PASS" if ok else "FAIL", name, detail))
    RESULTS.append(bool(ok))


def check_stage2(rng):
    print("\n5. stage 2: exact posterior and solver traces")
    import torch
    import hexagon_stage2 as H2
    from common import MeanFlowMLPLowFreq, VelocityMLPLowFreq, freeze
    ok, detail = True, []
    for sigma, y in ((0.1, 0.4), (0.3, -1.2)):
        post = H2.Posterior(y, sigma)
        x = H.sample(2_000_000, rng)
        w = np.exp(-0.5 * (y - x[:, 0]) ** 2 / sigma ** 2)
        w /= w.sum()
        is_mean = w @ x
        se = np.sqrt((w[:, None] ** 2 * (x - is_mean) ** 2).sum(0))
        smp = post.sample(200_000, rng)
        gap = float(np.max(np.abs(post.mean() - is_mean) / se))
        off = float(np.abs(smp.mean(0) - post.mean()).max())
        ok &= abs(post.w.sum() - 1) < 1e-12 and gap < 4 and off < 0.01
        detail.append("sigma %g: |mean - IS| %.1f se, samples' mean off %.4f" % (sigma, gap, off))
        if sigma == 0.1:                         # two branches: upper and lower edges
            ok &= (smp[:, 1] > 1).mean() > 0.4 and (smp[:, 1] < -1).mean() > 0.4
    report("posterior by quadrature = importance sampling", ok, "; ".join(detail))

    torch.manual_seed(0)
    priors = [H2.FlowPrior(freeze(VelocityMLPLowFreq())), H2.MeanFlowPrior(freeze(MeanFlowMLPLowFreq()))]
    task = H2.ControlTask("corner", 6, 1, "cpu")
    small = {"rhso": {"NM": (2, 3)}, "rhso_suffix": {"NM": (2, 3)}, "dflow": {"num_opt_steps": 3},
             "pnp": {"num_pnp_steps": 3}, "flower": {"num_flower_steps": 3},
             "mpc_delta_t": {"num_mpc_steps": 2, "n_ctrl": 2}, "mpc_rhc": {"K": 2}}
    same, lengths = True, {}
    for spec in H2.specs_for(task):
        for prior in priors:
            if prior.family not in spec.families:
                continue
            cont = [{k: v[1] for k, v in spec.continuous.items()}]
            a, _ = H2.run_point(spec, prior, task.rows, small[spec.name], cont)
            tr = []
            b, _ = H2.run_point(spec, prior, task.rows, small[spec.name], cont, trace=tr)
            same &= bool(np.array_equal(a, b)) and bool(np.allclose(tr[-1].double().cpu().numpy(), b))
            lengths[spec.name] = len(tr)
    want = {"rhso": 3, "rhso_suffix": 3, "dflow": 2, "pnp": 5, "flower": 4, "mpc_delta_t": 3, "mpc_rhc": 5}
    report("trace changes nothing", same and lengths == want,
           "identical outputs for every solver and prior; states recorded %s" % lengths)


def main():
    print("\n1. the hexagon")
    V = H.VERTICES
    sides = np.linalg.norm(np.roll(V, -1, 0) - V, axis=1)
    radii = np.linalg.norm(V, axis=1)
    report("regular, side 2, centred", np.allclose(sides, 2) and np.allclose(radii, 2)
           and np.allclose(V.mean(0), 0), "6 sides of %.6f, circumradius 2" % sides.mean())
    report("flat lower edge, lower-right corner (1, -sqrt 3)",
           np.allclose(V[4], [-1, -math.sqrt(3)]) and np.allclose(H.LOWER_RIGHT_CORNER, [1, -math.sqrt(3)]),
           "lower edge from (-1, -1.732) to (1, -1.732)")

    print("\n2. distance to the hexagon")
    apothem = math.sqrt(3)
    pts = np.array([V[0], V[3], (V[4] + V[5]) / 2, [0, 0], [0, -apothem - 0.5],
                    V[5] + 0.3 * V[5] / 2])
    want = np.array([0, 0, 0, apothem, 0.5, 0.3])
    got = H.distance_to_hexagon(pts)
    report("exact on known points", np.allclose(got, want, atol=1e-12),
           "vertices, midpoint, centre (sqrt 3), 0.5 below the lower edge, 0.3 beyond a corner")
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, (2000, 2))
    dense = H.boundary_point(np.linspace(0, H.PERIMETER, 120_001)[:-1])
    brute = np.min(np.linalg.norm(x[:, None] - dense[None], axis=2), axis=1)
    report("agrees with a dense polyline", np.abs(H.distance_to_hexagon(x) - brute).max() < 1e-4,
           "max difference %.1e on 2000 random points" % np.abs(H.distance_to_hexagon(x) - brute).max())

    print("\n3. the data")
    s = rng.uniform(0, H.PERIMETER, 600_000)
    edge_share = np.bincount(np.minimum((s // 2).astype(int), 5), minlength=6) / len(s)
    x = H.sample(600_000, rng)
    clean = H.sample(10_000, rng, jitter=0.0)
    angle = np.arctan2(x[:, 1], x[:, 0])
    sector = np.bincount(((angle % (2 * math.pi)) // (math.pi / 3)).astype(int) % 6, minlength=6) / len(x)
    report("uniform in arc length, centred", np.abs(edge_share - 1 / 6).max() < 0.003
           and np.abs(sector - 1 / 6).max() < 0.003 and np.abs(x.mean(0)).max() < 0.01
           and H.distance_to_hexagon(clean).max() < 1e-12,
           "edge shares %.4f..%.4f; max |mean| %.4f" % (sector.min(), sector.max(), np.abs(x.mean(0)).max()))
    d = H.distance_to_hexagon(x)
    mid = np.abs(((s % 2) - 1)) < 0.8                       # draws far from the corners
    xm = H.boundary_point(s[mid]) + H.JITTER * rng.standard_normal((mid.sum(), 2))
    dm = H.distance_to_hexagon(xm)
    report("distance follows the jitter", abs(dm.mean() - H.JITTER * math.sqrt(2 / math.pi)) < 3e-4
           and np.percentile(d, 99) < 3 * H.JITTER,
           "mean %.5f vs 0.02 sqrt(2/pi) = %.5f away from corners; p99 %.4f overall"
           % (dm.mean(), H.JITTER * math.sqrt(2 / math.pi), np.percentile(d, 99)))

    print("\n4. sliced W2")
    a = rng.standard_normal((20_000, 2))
    b = rng.standard_normal((20_000, 2)) + np.array([1.0, 0.0])
    sw = H.sliced_w2(a, b, n_proj=2000)
    report("N(0, I) vs N(e1, I)", abs(sw - 1 / math.sqrt(2)) < 0.02,
           "%.4f, exact |mu| / sqrt(2) = %.4f" % (sw, 1 / math.sqrt(2)))

    check_stage2(rng)
    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
