from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('aedf,bfce->abcd', A,B,semiring='+,*')

    return C

def test_mm_SemiringPlusTimes_dense_4Dtensors():
    A = np.full((2,2,2,2), 2.2)
    B = np.full((2,2,2,2), 3.6)

    res = run_comet(A,B)
    expected = np.array([31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68,31.68]).reshape((2,2,2,2))
    np.testing.assert_almost_equal(res, expected)