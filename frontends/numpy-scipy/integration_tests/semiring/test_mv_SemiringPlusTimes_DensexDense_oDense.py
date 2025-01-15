from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,j->i', A,B,semiring='+,*')

    return C

def test_mv_SemiringPlusTimes_DensexDense_oDense():
    A = np.full((8,16), 2.3)
    B = np.full((16), 3.7)

    res = run_comet(A,B)
    expected = np.array([136.16,136.16,136.16,136.16,136.16,136.16,136.16,136.16]).reshape((8))
    np.testing.assert_almost_equal(res, expected)