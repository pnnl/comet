from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,jk->ik', A,B,semiring='min,+')

    return C

def test_mm_SemiringMinPlus_CSRxCSR_oCSR(data_rank2_path):
    A = sp.sparse.csr_array(sp.io.mmread(data_rank2_path))
    B = sp.sparse.csr_array(sp.io.mmread(data_rank2_path))

    res = run_comet(A,B)
    expected = sp.sparse.csr_array(([2,2.4,4,4.5,6,5.1,5.5,7.2,7.7], [0,3,1,4,2,0,3,1,4], [0,2,4,5,7,9]))
    np.testing.assert_almost_equal(res.todense(), expected.todense())