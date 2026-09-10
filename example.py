from pathlib import Path
import csv, json
import numpy as np
from cavity_qed import simulate


def main(out=Path("results")):
    out.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 20, 1001)
    result = simulate(t, g=1.0, kappa=0.1, gamma=0.02)
    with (out / "populations.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["time", "emitter", "cavity", "trace"])
        w.writerows(zip(*[result[k] for k in ("time", "emitter", "cavity", "trace")]))
    rows = []
    for kappa in (0.2, 0.5, 1.0):
        r = simulate(np.linspace(0, 40, 2001), g=1.0, kappa=kappa, gamma=0.02)
        expected = 0.02 + 4 * 1.0**2 / kappa
        fit = np.linspace(5, 30, len(r["time"]))
        rate = -np.polyfit(fit, np.log(np.maximum(r["emitter"], 1e-12)), 1)[0]
        rows.append({"kappa": kappa, "measured_rate": float(rate), "relative_rate_error": float(abs(rate - expected) / expected)})
    (out / "purcell_rates.csv").write_text("kappa,measured_rate,relative_rate_error\n" + "\n".join(f'{r["kappa"]},{r["measured_rate"]},{r["relative_rate_error"]}' for r in rows) + "\n")
    (out / "metrics.json").write_text(json.dumps({"purcell": rows}, indent=2))
    import matplotlib.pyplot as plt
    plt.plot(t, result["emitter"], label="emitter")
    plt.plot(t, result["cavity"], label="cavity")
    plt.xlabel("time"); plt.ylabel("population"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "dynamics.png", dpi=120); plt.close()
