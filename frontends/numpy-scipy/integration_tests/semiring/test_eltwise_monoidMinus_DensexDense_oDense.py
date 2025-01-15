from cometpy import comet
import scipy as sp
import numpy as np

@comet.compile(flags=None)
def run_comet(A,B):
    C = comet.einsum('ab,ab->ab', A,B,semiring='-')

    return C


A = np.full((4,4), 4.2)
B = np.full((4,4), 2.7)

def test_eltwise_monoidMinus_DensexDense_oDense():
    res = run_comet(A,B)
    expected = np.array([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]).reshape((4,4))
    np.testing.assert_almost_equal(res, expected)