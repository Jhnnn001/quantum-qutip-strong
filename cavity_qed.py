"""Resonant Jaynes-Cummings dynamics from an excited emitter and empty cavity."""

import numpy as np
import qutip as qt


def simulate(times, *, g=1.0, kappa=0.0, gamma=0.0, cutoff=2):
    """Return populations and trace; all rates share one inverse-time unit.

    H = g(a†σ− + aσ+), with collapse operators sqrt(kappa)a and
    sqrt(gamma)σ−. Thus kappa and gamma are population decay rates.
    The cavity cutoff counts Fock states, including vacuum.
    """
    t = np.asarray(times, dtype=float)
    if (t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t))
            or t[0] != 0 or np.any(np.diff(t) <= 0)):
        raise ValueError("times must start at zero and increase strictly, with at least two finite values")
    rates = np.asarray([g, kappa, gamma], dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("g, kappa, gamma must be finite nonnegative rates")
    if not isinstance(cutoff, (int, np.integer)) or cutoff < 2:
        raise ValueError("cutoff must be an integer of at least two")

    a = qt.tensor(qt.destroy(cutoff), qt.qeye(2))
    sm = qt.tensor(qt.qeye(cutoff), qt.destroy(2))
    hamiltonian = g * (a.dag() * sm + a * sm.dag())
    initial = qt.tensor(qt.basis(cutoff, 0), qt.basis(2, 1))
    collapse = [np.sqrt(rate) * op for rate, op in [(kappa, a), (gamma, sm)] if rate > 0]
    solved = qt.mesolve(
        hamiltonian, initial, t, collapse,
        e_ops=[sm.dag() * sm, a.dag() * a, qt.tensor(qt.qeye(cutoff), qt.qeye(2))],
        options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 10000, "normalize_output": False},
    )
    return dict(zip(("time", "emitter", "cavity", "trace"), [t, *map(np.asarray, solved.expect)]))
