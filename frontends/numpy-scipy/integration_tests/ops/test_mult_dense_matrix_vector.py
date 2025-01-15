import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A,B):
	C = np.einsum('ij,j->i', A,B)

	return C

@comet.compile(flags=None)
def run_comet_with_jit(A,B):
	C = comet.einsum('ij,j->i', A,B)

	return C

def test_mult_dense_matrix_vector():
	A = np.full([8, 16], 2.3,  dtype=float)
	B = np.full([16], 3.7,  dtype=float)
	C = np.full([8], 0.0,  dtype=float)
	expected_result = run_numpy(A,B)
	result_with_jit = run_comet_with_jit(A,B)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
