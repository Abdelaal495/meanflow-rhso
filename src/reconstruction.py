"""The reconstruction-method registry: (dynamics family, method) -> reconstructor.

This module owns the GLOBAL dispatch and nothing else.  It makes no experimental decision,
holds no hyperparameter and knows nothing about any individual algorithm beyond its
signature; each strategy lives in its own module:

    sdedit.py   ordinary generation from z_t0            (the paired baseline)
    mpc.py      MPC-RHC and MPC-delta_t                  (trajectory control)
    pnp.py      PnP-Flow                                 (gradient / reproject / denoise)
    dflow.py    D-Flow                                   (source-state optimisation)
    rhso.py     Receding-Horizon State Optimization      (state optimisation + replanning)
    flower.py   Flower                                   (destination / proximal refinement /
                                                          re-noising; pixel priors only)

Every reconstructor has the same signature (see docs/extending.md):

    fn(adapter, cond, x0, problem, spec) -> (final native state, ReconstructionStats)

`x0` always arrives already built by `models.base.build_initial_state`, from the shared
epsilon, so the fairness invariant

    same problem + same model + same t0 + same epsilon
        ==>  the ONLY thing that varies is the reconstruction strategy

holds for every entry in the table below by construction rather than by convention.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .utils import MEANFLOW, STANDARD_FLOW

# (dynamics_family, method) -> callable
RECONSTRUCTORS: Dict[Tuple[str, str], Callable[..., Any]] = {}


def register_reconstructor(dynamics_family: str, method: str, fn: Callable[..., Any],
                           overwrite: bool = False) -> Callable[..., Any]:
    """Register one strategy for one dynamics family."""
    if dynamics_family not in (STANDARD_FLOW, MEANFLOW):
        raise ValueError("Unknown dynamics family %r" % (dynamics_family,))
    key = (dynamics_family, method)
    if key in RECONSTRUCTORS and not overwrite:
        raise ValueError("A reconstructor is already registered for %r / %r"
                         % (dynamics_family, method))
    RECONSTRUCTORS[key] = fn
    return fn


def _build_registry() -> None:
    """Populate the table from the algorithm modules.

    Imported here rather than at module scope of each algorithm file so that the
    dependency runs one way: reconstruction.py knows about the algorithms, and no
    algorithm module has to know about the registry.
    """
    if RECONSTRUCTORS:
        return
    from .dflow import flow_dflow, meanflow_dflow
    from .flower import flow_flower, meanflow_flower
    from .mpc import flow_mpc_delta_t, flow_mpc_rhc, meanflow_mpc_delta_t, meanflow_mpc_rhc
    from .pnp import flow_pnp, meanflow_pnp
    from .rhso import flow_rhso, meanflow_rhso
    from .sdedit import sdedit_flow, sdedit_meanflow

    register_reconstructor(STANDARD_FLOW, "sdedit", _drop_problem(sdedit_flow))
    register_reconstructor(MEANFLOW, "sdedit", _drop_problem(sdedit_meanflow))

    register_reconstructor(STANDARD_FLOW, "mpc_rhc", flow_mpc_rhc)
    register_reconstructor(STANDARD_FLOW, "mpc_delta_t", flow_mpc_delta_t)
    register_reconstructor(MEANFLOW, "mpc_rhc", meanflow_mpc_rhc)
    register_reconstructor(MEANFLOW, "mpc_delta_t", meanflow_mpc_delta_t)

    register_reconstructor(STANDARD_FLOW, "pnp", flow_pnp)
    register_reconstructor(MEANFLOW, "pnp", meanflow_pnp)

    register_reconstructor(STANDARD_FLOW, "dflow", flow_dflow)
    register_reconstructor(MEANFLOW, "dflow", meanflow_dflow)

    register_reconstructor(STANDARD_FLOW, "rhso", flow_rhso)
    register_reconstructor(MEANFLOW, "rhso", meanflow_rhso)

    # Both families are registered; the capability table limits Flower to the pixel priors
    # (pMF, JiT), and flower.py itself refuses a latent adapter.
    register_reconstructor(STANDARD_FLOW, "flower", flow_flower)
    register_reconstructor(MEANFLOW, "flower", meanflow_flower)


def _drop_problem(fn):
    """Adapt SDEdit's (adapter, cond, x0, spec) signature to the common one.

    SDEdit never sees the measurement -- that is the point of the baseline -- so its
    reconstructors genuinely do not take `problem`.  Dropping the argument here keeps the
    fact visible instead of hiding it behind an unused parameter.
    """
    def wrapper(adapter, cond, x0, problem, spec):        # noqa: ARG001  (problem unused)
        return fn(adapter, cond, x0, spec)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def select_reconstructor(dynamics_family: str, method: str):
    """The whole family/method dispatch, in one lookup."""
    _build_registry()
    try:
        return RECONSTRUCTORS[(dynamics_family, method)]
    except KeyError:
        known = sorted({m for (_f, m) in RECONSTRUCTORS})
        raise ValueError("No reconstruction strategy for %r / %r. Registered methods: %s"
                         % (dynamics_family, method, known))


def registered_methods() -> Dict[str, Tuple[str, ...]]:
    """method -> the dynamics families that implement it (used by the checks)."""
    _build_registry()
    out: Dict[str, Tuple[str, ...]] = {}
    for family, method in RECONSTRUCTORS:
        out[method] = tuple(sorted(set(out.get(method, ())) | {family}))
    return out



