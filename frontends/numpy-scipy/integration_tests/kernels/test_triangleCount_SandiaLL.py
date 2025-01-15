import time
import numpy as np
import scipy as sp
from cometpy import comet

def run_numpy(L0,L1,L2):
	C =  ((L1 @ L2) * L0).sum()
	return C

@comet.compile(flags=None)
def run_comet_with_jit(L0,L1,L2):
	C =  ((L1 @ L2) * L0).sum()
	return C

def test_triangleCount_SandiaLL(data_tc_path):
	A = sp.sparse.csr_array(sp.io.mmread(data_tc_path))
	L0 = sp.sparse.csr_array(sp.sparse.tril(A, format='csr'))
	expected_result = run_numpy(L0,L0,L0)
	result_with_jit = run_comet_with_jit(L0,L0,L0)
	if sp.sparse.issparse(expected_result):
		expected_result = expected_result.todense()
	if sp.sparse.issparse(result_with_jit):
		result_with_jit = result_with_jit.todense()
	np.testing.assert_almost_equal(result_with_jit, expected_result)
