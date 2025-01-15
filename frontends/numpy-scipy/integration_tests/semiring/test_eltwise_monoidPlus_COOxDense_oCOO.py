from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,ij->ij', A,B,semiring='+')

    return C

def test_eltwise_monoidPlus_COOxDense_oCOO(data_rank2_path):
    A = sp.sparse.coo_array(sp.io.mmread(data_rank2_path))
    B = np.full((A.shape), 2.7)

    res = run_comet(A,B)
    expected = sp.sparse.coo_array(([3.7,4.1,4.7,5.2,5.7,6.8,6.7,7.9,7.7], ([0,0,1,1,2,3,3,4,4], [0,3,1,4,2,0,3,1,4])))
    np.testing.assert_almost_equal(res.todense(), expected.todense())