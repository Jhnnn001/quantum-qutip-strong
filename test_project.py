"""Run with python test_project.py; checks use analytic physical limits."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory

assert importlib.util.find_spec("cavity_qed") is not None, "Implement cavity_qed.py first"

import numpy as np

from cavity_qed import simulate


def main():
    t = np.linspace(0, 12, 601)
    lossless = simulate(t)
    exchange_error = max(
        np.max(np.abs(lossless["emitter"] - np.cos(t) ** 2)),
        np.max(np.abs(lossless["cavity"] - np.sin(t) ** 2)),
    )
    assert exchange_error < 2e-7, exchange_error
    assert np.max(np.abs(lossless["emitter"] + lossless["cavity"] - 1)) < 2e-7

    uncoupled = simulate(t, g=0, gamma=0.3, kappa=0.8)
    decay_error = np.max(np.abs(uncoupled["emitter"] - np.exp(-0.3 * t)))
    assert decay_error < 2e-8, decay_error
    assert np.max(np.abs(uncoupled["cavity"])) < 1e-12

    cutoff_error = 0.0
    for kappa, gamma in [(0, 0), (0.2, 0.05), (20, 0.02)]:
        result = simulate(t, kappa=kappa, gamma=gamma)
        larger = simulate(t, kappa=kappa, gamma=gamma, cutoff=3)
        for name in ("emitter", "cavity"):
            assert result[name].min() > -2e-8
            assert result[name].max() < 1 + 2e-8
            cutoff_error = max(cutoff_error, np.max(np.abs(result[name] - larger[name])))
        assert np.max(np.abs(result["trace"] - 1)) < 2e-7
        assert np.max(np.diff(result["emitter"] + result["cavity"])) < 2e-8
    assert cutoff_error < 2e-7, cutoff_error

    # At g=1, kappa=80, gamma=0.02, adiabatic elimination predicts 0.07.
    t_bad = np.linspace(0, 60, 1201)
    bad = simulate(t_bad, kappa=80, gamma=0.02)
    fit = (t_bad >= 0.125) & (t_bad <= 40)
    measured_rate = -np.polyfit(t_bad[fit], np.log(bad["emitter"][fit]), 1)[0]
    assert abs(measured_rate / 0.07 - 1) < 0.002, measured_rate

    for times, parameters in [
        ([0, 1], {"kappa": -1}), ([0, 1], {"g": float("nan")}),
        ([0, 1], {"cutoff": 1}), ([0, 1], {"cutoff": 2.5}),
        ([1, 2], {}), ([0, 0], {}), ([0, float("nan")], {}),
        ([[0, 1]], {}), ([0], {}),
    ]:
        try:
            simulate(times, **parameters)
        except ValueError:
            pass
        else:
            raise AssertionError((times, parameters))

    print(f"PASS: analytic exchange max error {exchange_error:.3g}")
    print(f"PASS: uncoupled decay max error {decay_error:.3g}")
    print(f"PASS: cutoff 2 versus 3 max population difference {cutoff_error:.3g}")
    print(f"PASS: trace, populations, loss monotonicity, input validation")
    print(f"PASS: bad-cavity rate {measured_rate:.8g}; prediction 0.07")

    assert importlib.util.find_spec("example") is not None, "Implement example.py first"
    from example import main as run_example

    with TemporaryDirectory() as folder:
        run_example(Path(folder))
        metrics = json.loads((Path(folder) / "metrics.json").read_text())
        rates = [row["measured_rate"] for row in metrics["purcell"]]
        assert all(np.isfinite(r) and r > 0 for r in rates), rates
        for name in ("populations.csv", "purcell_rates.csv", "dynamics.png"):
            assert (Path(folder) / name).stat().st_size > 100
    print("PASS: reproducible example outputs and converging Purcell-rate comparison")


if __name__ == "__main__":
    main()
