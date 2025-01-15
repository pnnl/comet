from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,j->i', A,B,semiring='+,*')

    return C

def test_mv_SemiringPlusTimes_COOxDense_oDense(data_rank2_path):
    A = sp.sparse.coo_array(sp.io.mmread(data_rank2_path))
    B = np.full((A.shape[1]), 1.7)

    res = run_comet(A,B)
    expected = np.array([4.08,7.65,5.1,13.77,17.34]).reshape((A.shape[0]))
    np.testing.assert_almost_equal(res, expected)