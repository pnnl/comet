import time
import numpy as np
import scipy as sp
from cometpy import comet
def run_numpy(A,B):
	C = np.einsum('ij,jk->ik', A,B)

	return C

@comet.compile(flags=None, target="gpu")
def run_comet_with_jit(A,B):
	C = comet.einsum('ij,jk->ik', A,B)

	return C

A = np.full([8, 4], 2.2,  dtype=float)
B = np.full([4, 2], 3.4,  dtype=float)
C = np.full([8, 2], 0.0,  dtype=float)
expected_result = run_numpy(A,B)
result_with_jit = run_comet_with_jit(A,B)
if sp.sparse.issparse(expected_result):
	expected_result = expected_result.todense()
	result_with_jit = result_with_jit.todense()
np.testing.assert_almost_equal(result_with_jit, expected_result)
