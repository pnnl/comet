import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A):
	B = A.transpose()

	return B

@comet.compile(flags=None, target="gpu")
def run_comet_with_jit(A):
	B = A.transpose()

	return B

A = np.full([4, 4], 3.2,  dtype=float)
expected_result = run_numpy(A)
result_with_jit = run_comet_with_jit(A)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)
