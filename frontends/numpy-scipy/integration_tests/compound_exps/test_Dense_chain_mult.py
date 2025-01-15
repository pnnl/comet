from cometpy import comet
import numpy as np

@comet.compile(flags="")
def comet_chain_multiply(A,B,C,D):
    # res = comet.einsum('ij,jk,kl,lm->im', A, C, B, C.transpose()) @ D
    res = A @ C @ B @ C.transpose() @ D

    return res

def numpy_chain_multiply(A,B,C,D):
    # res = np.einsum('ij,jk,kl,lm->im', A, C, B, C.transpose()) @ D
    res = A @ C @ B @ C.transpose() @ D


    return  res

def test_Dense_chain_mult():
    A = np.full([2,4], 1.0)
    B = np.full([4,4], 1.0)
    C = np.full([4,4], 1.0)
    D = np.full([4,6], 1.0)

    Dn = numpy_chain_multiply(A,B,C,D)
    Dc = comet_chain_multiply(A,B,C,D)

    np.testing.assert_almost_equal(Dn,Dc)