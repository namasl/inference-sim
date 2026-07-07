# campaigns/edpp-study/analyze/test_utilization_sweep.py
"""Self-check for utilization_sweep.py aggregation, drift, and verdict math.
Run: python3 campaigns/edpp-study/analyze/test_utilization_sweep.py
Exits nonzero on failure. No external test framework (matches analyze/ convention)."""
import json, os, subprocess, sys, tempfile, csv

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "utilization_sweep.py")


def _write(path, text):
    with open(path, "w") as f:
        f.write(text)


def test_scan_verdict_stable_exit0():
    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "m.json")
        _write(m, json.dumps({"responses_per_sec": 1.0, "saturation": {"level": "STABLE"}}))
        r = subprocess.run([sys.executable, SCRIPT, "scan-verdict", "--metrics", m],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert "STABLE" in r.stdout


def test_scan_verdict_overloaded_exit1():
    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "m.json")
        _write(m, json.dumps({"responses_per_sec": 1.0, "saturation": {"level": "OVERLOADED"}}))
        r = subprocess.run([sys.executable, SCRIPT, "scan-verdict", "--metrics", m],
                           capture_output=True, text=True)
        assert r.returncode == 1


def _point(d, rate, respsec, level, realized, pred_waiting):
    # metrics.json
    _write(os.path.join(d, f"pt_rate{rate}.metrics.json"),
           json.dumps({"responses_per_sec": respsec, "saturation": {"level": level}}))
    # ablation.json (only the fields aggregate reads)
    _write(os.path.join(d, f"pt_rate{rate}.ablation.json"), json.dumps({
        "pools": {"local": {"estimators": {
            "waiting": {"median_ratio_real_over_pred": realized / pred_waiting,
                        "median_signed_error_us": realized - pred_waiting,
                        "realized_p50_us": realized},
            "rollforward": {"median_ratio_real_over_pred": 1.0,
                            "median_signed_error_us": 0.0, "realized_p50_us": realized},
        }}}}))
    # admission.csv: 8 rows, second half 2x the first half -> drift 2.0
    with open(os.path.join(d, f"pt_rate{rate}.admission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "pool", "realized_t_adm", "t_adm_pred_waiting"])
        vals = [10, 10, 10, 10, 20, 20, 20, 20]
        for i, v in enumerate(vals):
            w.writerow([f"request_{i}", "local", v, pred_waiting])


def test_aggregate_rho_drift_and_passthrough():
    with tempfile.TemporaryDirectory() as d:
        _point(d, "0.5", respsec=0.5, level="STABLE", realized=100.0, pred_waiting=50.0)
        out = os.path.join(d, "sweep.json")
        r = subprocess.run([sys.executable, SCRIPT, "aggregate", "--sweep-dir", d,
                            "--lambda-star", "1.0", "--pool", "local", "--out", out,
                            "--warmup", "0"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rep = json.load(open(out))
        pt = rep["points"][0]
        assert abs(pt["rho_hat"] - 0.5) < 1e-9              # 0.5 / 1.0
        assert abs(pt["admission_drift"] - 2.0) < 1e-9      # median(20)/median(10)
        assert pt["stationary_verdict"] == "STABLE"
        assert abs(pt["estimators"]["waiting"]["median_ratio_real_over_pred"] - 2.0) < 1e-9
        assert rep["lambda_star"] == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("all passed")
