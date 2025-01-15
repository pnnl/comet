import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A):
	B = np.einsum('ijkl->ikjl', A)

	return B

@comet.compile(flags="-opt-dense-transpose")
def run_comet_with_jit(A):
	B = comet.einsum('ijkl->ikjl', A)

	return B

def test_opt_dense_transpose(): 
	A = np.full([2, 4, 8, 16], 3.7,  dtype=float)
	expected_result = run_numpy(A)
	result_with_jit = run_comet_with_jit(A)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
