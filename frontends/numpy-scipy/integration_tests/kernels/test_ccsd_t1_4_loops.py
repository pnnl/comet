import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(v,t1):
	i0 = np.einsum('cima,mc->ia', v,t1)

	return i0

@comet.compile(flags=None)
def run_comet_with_jit(v,t1):
	i0 = comet.einsum('cima,mc->ia', v,t1)

	return i0

def test_ccsd_t1_4_loops():
	v = np.full([2, 2, 4, 4], 2.3,  dtype=float)
	t1 = np.full([4, 2], 3.4,  dtype=float)
	i0 = np.full([2, 4], 0.0,  dtype=float)
	expected_result = run_numpy(v,t1)
	result_with_jit = run_comet_with_jit(v,t1)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
