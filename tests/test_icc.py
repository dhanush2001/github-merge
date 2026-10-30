"""ICC columns of the crossed variance decomposition (Task 2 / reviewer W5)."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from variance_decomposition import LOGIT_RESIDUAL, icc_table  # noqa: E402

COMPS = {"scenario": 0.04, "admin": 0.01, "admin_x_dev": 0.001, "dev": 0.0002, "residual": 0.10}
LOGIT = {"scenario": 3.0, "admin": 1.5, "admin_x_dev": 0.05, "dev": 0.02}


def test_iccs_sum_to_one_on_both_scales():
    t = icc_table(COMPS, LOGIT)
    assert t.icc_linear.sum() == pytest.approx(1.0)
    assert t.icc_logit.sum() == pytest.approx(1.0)


def test_icc_definitions():
    t = icc_table(COMPS, LOGIT).set_index("component")
    assert t.loc["admin", "icc_linear"] == pytest.approx(0.01 / sum(COMPS.values()))
    assert t.loc["admin", "icc_logit"] == pytest.approx(1.5 / (sum(LOGIT.values()) + np.pi ** 2 / 3))
    assert t.loc["residual", "variance_logit"] == pytest.approx(LOGIT_RESIDUAL)
    assert np.isnan(t.loc["residual", "pct_of_random"])
    assert t.loc[["scenario", "admin", "admin_x_dev", "dev"], "pct_of_random"].sum() == pytest.approx(100)
