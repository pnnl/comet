from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,ij->ij', A,B,semiring='*')

    return C

def test_eltwise_monoidTimes_COOxDense_oCOO(data_rank2_path):
    A = sp.sparse.coo_array(sp.io.mmread(data_rank2_path))
    B = np.full((A.shape), 2.7)

    res = run_comet(A,B)
    expected = sp.sparse.coo_array(([2.7,3.78,5.4,6.75,8.1,11.07,10.8,14.04,13.5], ([0,0,1,1,2,3,3,4,4], [0,3,1,4,2,0,3,1,4])))
    np.testing.assert_almost_equal(res.todense(), expected.todense())