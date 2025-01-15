import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A,B):
	C = A @ B.transpose()

	return C

@comet.compile(flags=None)
def run_comet_with_jit(A,B):
	C = A @ B.transpose()

	return C

def test_Dense_mult_dTranspose():
	B = np.full([5, 5], 3.2,  dtype=float)
	A = np.full([5, 5], 2.3,  dtype=float)
	C = np.full([5, 5], 0.0,  dtype=float)
	expected_result = run_numpy(A,B)
	result_with_jit = run_comet_with_jit(A,B)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
