"""Hexagon study (rebuttal Task 4): the data and the metrics, with no trained model.

1. the hexagon: regular, side 2, centred, flat lower edge, lower-right corner (1, -sqrt 3);
2. the distance to the hexagon is exact on known points (vertices, edge midpoints, the
   centre, points outside an edge and beyond a corner) and agrees with a dense polyline;
3. the data: uniform in arc length (each edge gets 1/6), centred, and its distance to the
   hexagon follows the jitter (|N(0, 0.02^2 I)| projected on the edge normal, away from
   corners: mean 0.02 sqrt(2 / pi));
4. sliced W2 is right on a known pair of samples.
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

    print("\n%d/%d checks passed" % (sum(RESULTS), len(RESULTS)))
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
