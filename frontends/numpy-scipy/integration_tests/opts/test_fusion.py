import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(B,C,D):
	T = B @ C 
	A = T @ D

	return A

@comet.compile(flags="--opt-fusion")
def run_comet_with_jit(B,C,D):
	T = B @ C
	A = T @ D

	return A

def test_fusion(data_rank2_path):
	B = sp.sparse.csr_array(sp.io.mmread(data_rank2_path))
	C = np.full([B.shape[1], 4], 1.2,  dtype=float)
	D = np.full([4, 4], 3.4,  dtype=float)
	A = np.full([B.shape[0], 4], 0.0,  dtype=float)
	T = np.full([B.shape[0], 4], 0.0,  dtype=float)
	expected_result = run_numpy(B,C,D)
	result_with_jit = run_comet_with_jit(B,C,D)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
