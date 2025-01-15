import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(v,t2):
	i0 = np.einsum('icmn,mnca->ia', v,t2)

	return i0

@comet.compile(flags="--opt-matmul-tiling --convert-tc-to-ttgt")
def run_comet_with_jit(v,t2):
	i0 = comet.einsum('icmn,mnca->ia', v,t2)

	return i0

def test_ccsd_t1_21_ttgt_tilling():
	v = np.full([2, 2, 4, 4], 2.3,  dtype=float)
	t2 = np.full([4, 4, 2, 4], 3.4,  dtype=float)
	i0 = np.full([2, 4], 0.0,  dtype=float)
	expected_result = run_numpy(v,t2)
	result_with_jit = run_comet_with_jit(v,t2)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
