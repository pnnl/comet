from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ij,jk->ik', A,B,semiring='+,*')

    return C

def test_mm_SemiringPlusTimes_DensexDense_oDense():
    A = np.full((8, 4), 2.2)
    B = np.full((4, 2), 3.4)

    res = run_comet(A,B)
    expected = np.array([29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92,29.92]).reshape((8, 2))
    np.testing.assert_almost_equal(res, expected)