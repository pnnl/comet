from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ab,ab->ab', A,B,semiring='*')

    return C

def test_eltwise_monoidTimes_DensexDense_oDense():
    A = np.full((4,4), 2.7)
    B = np.full((4,4), 3.2)

    res = run_comet(A,B)
    expected = np.array([8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64,8.64]).reshape((4,4))
    np.testing.assert_almost_equal(res, expected)