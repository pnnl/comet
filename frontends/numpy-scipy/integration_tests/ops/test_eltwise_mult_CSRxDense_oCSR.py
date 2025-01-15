import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(A,B):
	C = A.multiply( B) 

	return C

@comet.compile(flags=None)
def run_comet_with_jit(A,B):
	C = A.multiply( B) 

	return C


def test_eltwise_mult_CSRxDense_oCSR(data_rank2_path): 
	A = sp.sparse.csr_array(sp.io.mmread(data_rank2_path))
	B = np.full([A.shape[0], A.shape[1]], 2.7,  dtype=float)
	expected_result = run_numpy(A,B)
	result_with_jit = run_comet_with_jit(A,B)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
	if sp.sparse.issparse(result_with_jit):
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
