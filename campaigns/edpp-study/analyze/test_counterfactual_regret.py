# campaigns/edpp-study/analyze/test_counterfactual_regret.py
"""Self-check for counterfactual_regret.py. Run: python3 .../test_counterfactual_regret.py"""
import json, os, subprocess, sys, tempfile, csv

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "counterfactual_regret.py")


def test_capture_plan_from_outcome():
    with tempfile.TemporaryDirectory() as d:
        oc = os.path.join(d, "outcome.csv")
        with open(oc, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["request_id", "disaggregated", "prefill_instance", "decode_instance"])
            w.writerow(["r1", "false", "", "M0"]); w.writerow(["r2", "true", "P0", "M1"])
        out = os.path.join(d, "plan.csv")
        r = subprocess.run([sys.executable, SCRIPT, "capture-plan", "--outcome", oc, "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rows = list(csv.DictReader(open(out)))
        assert rows[0] == {"request_id": "r1", "decode_instance": "M0", "prefill_instance": "local"}
        assert rows[1] == {"request_id": "r2", "decode_instance": "M1", "prefill_instance": "P0"}


def test_regret_aggregation():
    with tempfile.TemporaryDirectory() as d:
        def m(path, att): json.dump({"slo_attainment": att}, open(path, "w"))
        m(os.path.join(d, "baseline.json"), 0.80)
        # r1: one deviation improves to 0.90 -> regret 0.10; another 0.70
        m(os.path.join(d, "dev_r1_M1-local.json"), 0.90)
        m(os.path.join(d, "dev_r1_M0-P0.json"), 0.70)
        # r2: no deviation beats baseline -> regret 0
        m(os.path.join(d, "dev_r2_M0-local.json"), 0.75)
        out = os.path.join(d, "regret.json")
        r = subprocess.run([sys.executable, SCRIPT, "regret", "--sweep-dir", d, "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rep = json.load(open(out))
        by = {p["request_id"]: p for p in rep["per_request"]}
        assert abs(by["r1"]["regret"] - 0.10) < 1e-9 and by["r1"]["hindsight_best"] == "M1-local"
        assert abs(by["r2"]["regret"] - 0.0) < 1e-9
        assert abs(rep["total_regret"] - 0.10) < 1e-9 and rep["frac_positive_regret"] == 0.5


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok ", name)
    print("all passed")
