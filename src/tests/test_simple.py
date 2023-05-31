# src/tests/test_simple.py
import pytest
from .. import train
from .. import test
from .. import preprocess

@pytest.fixture()
def var_smoothing1():
    var_smoothing1 = 1.1e-09    
    yield var_smoothing1

@pytest.fixture()
def smoothing1_None_randomstate_score(var_smoothing1):
    # var_smoothing1 = 1.1e-09    
    preprocess.test_preprocess(random_state=None)
    train.test_train(var_smoothing1, None)
    score1 = test.test()
    # baseline score
    yield score1
    

def test_model_diff(smoothing1_None_randomstate_score):
    # var_smoothing1 = 1.1e-09
    var_smoothing2 = 1.2e-09
    train.test_train(var_smoothing2, None)
    score2 = test.test()
    accuracy_diff = abs(smoothing1_None_randomstate_score - score2)
    assert (
        accuracy_diff < 0.1
    ), f"Difference in model accuracy with different seeds is too high: {accuracy_diff}"

def test_nondeterminism_robustness(smoothing1_None_randomstate_score, var_smoothing1):
    for seed in [1, 2, 3, 4, 5, 42]:
        preprocess.test_preprocess(random_state=seed)
        train.test_train(var_smoothing1, seed)
        score42 = test.test(random_state=seed)
        accuracy_diff = abs(smoothing1_None_randomstate_score - score42)
        assert (
            accuracy_diff < 0.1
        ), f"Difference in model accuracy with different seeds is too high: {accuracy_diff}, seed: {seed}"