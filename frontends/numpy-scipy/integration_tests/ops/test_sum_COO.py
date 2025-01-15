import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A):
	var = A.sum()

	return var

@comet.compile(flags=None)
def run_comet_with_jit(A):
	var = A.sum()

	return var

def test_sum_COO(data_rank2_path):
	A = sp.sparse.coo_array(sp.io.mmread(data_rank2_path))
	expected_result = run_numpy(A)
	result_with_jit = run_comet_with_jit(A)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
