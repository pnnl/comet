import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy():
    a = 5 + 1
    b = a + 5 + 1
    c = b / 2
    d = c * 3
    e = d - 1

    return e

@comet.compile(flags=None)
def run_comet_with_jit():
    a = 5 + 1
    b = a + 5 + 1
    c = b / 2
    d = c * 3
    e = d - 1

    return e

expected_result = run_numpy()
result_with_jit = run_comet_with_jit()
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)