import itertools
import os

import pandas as pd
import pytest

FILES = ["creditcard.csv", "global_test.csv"] + [f"bank_{i}.csv" for i in range(4)]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(f"data/{f}") for f in FILES),
    reason="run partition.py first",
)


@pytest.fixture(scope="module")
def frames():
    full = pd.read_csv("data/creditcard.csv")
    test = pd.read_csv("data/global_test.csv")
    banks = [pd.read_csv(f"data/bank_{i}.csv") for i in range(4)]
    return full, test, banks


def test_no_rows_lost_or_duplicated(frames):
    full, test, banks = frames
    assert len(test) + sum(len(b) for b in banks) == len(full)


def test_fraud_cases_conserved(frames):
    full, test, banks = frames
    assert test["Class"].sum() + sum(b["Class"].sum() for b in banks) == full["Class"].sum()


def test_test_set_is_stratified(frames):
    full, test, _ = frames
    assert test["Class"].mean() == pytest.approx(full["Class"].mean(), abs=1e-4)

def test_banks_are_non_iid_by_amount(frames):
    _, _, banks = frames
    for a, b in itertools.pairwise(banks):
        assert a["Amount"].max() <= b["Amount"].min()