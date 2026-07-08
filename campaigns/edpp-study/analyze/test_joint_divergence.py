# campaigns/edpp-study/analyze/test_joint_divergence.py
"""Self-check for joint_divergence.py. Run: python3 .../test_joint_divergence.py

Also runnable via the script's own subcommand: `joint_divergence.py selftest`.
Plain-assert, no framework (matches test_counterfactual_regret.py)."""
import importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("joint_divergence", os.path.join(HERE, "joint_divergence.py"))
jd = importlib.util.module_from_spec(spec); spec.loader.exec_module(jd)


def test_divergence_rates_and_direction():
    # r0: d diverges, joint strictly lower J   -> lower_J
    # r1: agree on both                         -> not divergent
    # r2: p diverges, both J == 0 (tie-break)   -> tie_J, disagg
    # r3: d diverges, j_joint == j_scorer nonzero-> tie_J
    rows = [
        {"request_id": "r0", "agree_d": "false", "agree_p": "true", "j_scorer": "0.01", "j_joint": "0.004", "disaggregate": "false"},
        {"request_id": "r1", "agree_d": "true", "agree_p": "true", "j_scorer": "0.02", "j_joint": "0.02", "disaggregate": "false"},
        {"request_id": "r2", "agree_d": "true", "agree_p": "false", "j_scorer": "0", "j_joint": "0", "disaggregate": "true"},
        {"request_id": "r3", "agree_d": "false", "agree_p": "true", "j_scorer": "0.05", "j_joint": "0.05", "disaggregate": "false"},
    ]
    rep = jd.summarize(rows)
    assert rep["n"] == 4
    assert rep["counts"]["d_div"] == 2 and rep["counts"]["p_div"] == 1
    assert rep["n_divergent"] == 3
    assert abs(rep["d_divergence_rate"] - 0.5) < 1e-12
    assert abs(rep["any_divergence_rate"] - 0.75) < 1e-12
    assert rep["counts"]["lower_J"] == 1 and rep["counts"]["tie_J"] == 2 and rep["counts"]["higher_J"] == 0
    d = rep["direction_on_divergent"]
    assert abs(d["dir_lower_J"] - 1 / 3) < 1e-12 and abs(d["dir_tie_J"] - 2 / 3) < 1e-12
    assert abs(d["disagg_share"] - 1 / 3) < 1e-12


def test_float_dust_is_a_tie_not_a_higher_J_violation():
    # j_scorer == 0 exactly, j_joint == 1e-21 (summation dust): must classify as a
    # tie, NOT a spurious argmin-invariant violation (dir_higher_J must stay 0).
    rows = [{"request_id": "r", "agree_d": "false", "agree_p": "true",
             "j_scorer": "0", "j_joint": "1e-21", "disaggregate": "false"}]
    rep = jd.summarize(rows)
    assert rep["counts"]["higher_J"] == 0, rep
    assert rep["counts"]["tie_J"] == 1, rep


def test_empty_trace():
    rep = jd.summarize([])
    assert rep["n"] == 0 and rep["any_divergence_rate"] == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok ", name)
    print("all passed")
