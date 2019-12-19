from scipy.io import loadmat
import sys
import numpy as np
from numpy.fft import *
from scipy.sparse import *
from pylab import figure, cm
from matplotlib.colors import LogNorm
import warnings

def high_resolution_spectogram(x, q, tdeci, over, noct, minf, maxf):
    """Create time-frequency representation

    Python implementation of Linear Reassignment as outlined in
        Sparse time-frequency representations by Timothy J. Gardner and Marcelo O. Magnasco.
    Code in MATLAB by authors of the paper can be found here:
        https://github.com/earthspecies/spectral_hyperresolution/blob/master/reassignmentgw.m

    Args:
        x (numpy.ndarray of shape (N, C)):
            signal, an array of sampled amplitudes with values on the interval from -1 to 1,
            where N is the number of samples and C number of channels
        q (float):
            the Q of a wavelet
            good values to try when working with tonal sounds are 2, 4, 8 and 1 and 0.5 for impulsive sounds
        tdeci (int): temporal stride in samples, bigger means narrower picture
        over (int):  number of frequencies tried per vertical pixel
        minf (float):  the smallest frequency to visualize
        maxf (float):  the largest frequency to visualize

        natural units: time in samples, frequency [0,1) where 1=sampling rate

        For a more indepth treatment of the parameters please see:
            https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example_in_Python.ipynb
    """
    assert x.ndim == 2, 'signal (x) has to be two dimensional'

    lint = 0.5         # do not follow if reassignment takes you far
    eps = 1e-20

    noct = np.array([[noct]])
    over = np.array([[over]])
    tdeci = np.array([[tdeci]])
    minf = np.array([[minf]])
    maxf = np.array([[maxf]])
    N = np.array([[x.shape[0]]])
    xf = fftn(x, axes=[0])

    HT = np.ceil(N/tdeci).astype(np.int)
    HF = np.ceil(-noct*np.log2(minf/maxf)+1).astype(np.int)
    f = np.arange(0, N) / N
    f[f>0.5]=f[f>0.5]-1

    minf = minf.astype(np.float64)
    maxf = maxf.astype(np.float64)
    over = over.astype(np.float64)
    noct = noct.astype(np.float32)

    histo = np.zeros((HT[0][0], HF[0][0]))
    histc = np.zeros((HT[0][0], HF[0][0]))

    for log2f0 in np.arange(0, HF.astype(np.float32)*over):
        f0 = minf*2**(log2f0/over/noct)
        sigma = f0/(2*np.pi*q)
        gau = np.exp(-(f-f0)**2 / (2*sigma**2))
        gde = -1/sigma**1 * (f-f0) * gau

        xi = ifftn(gau.T * xf, axes=[0])
        eta = ifftn(gde.T * xf, axes=[0])
        mp = eta / (xi + eps)
        ener = abs(xi)**2

        tins = np.arange(1, N+1).reshape(-1, 1) + np.imag(mp)/(2*np.pi*sigma)
        fins = f0 - np.real(mp)*sigma;

        mask = (abs(mp)<lint) & (fins < maxf) & (fins>minf) & (tins>=1) & (tins<N)

        tins = tins[mask]
        fins = fins[mask]
        ener = ener[mask]

        itins = np.round(tins/tdeci+0.5)
        ifins = np.round(-noct*np.log2(fins/maxf)+1)

        np.add.at(histo, (itins[0].astype(int)-1, ifins[0].astype(int)-1), ener)
        np.add.at(histc, (itins[0].astype(int)-1, ifins[0].astype(int)-1), 0*itins[0]+1)

    mm = histc.max()
    histo[histc < np.sqrt(mm)] = 0

    return histo

def high_resolution_spectogram_sparse(x, q, tdeci, over, noct, minf, maxf):
    """Create time-frequency representation

    Python implementation of Linear Reassignment as outlined in
        Sparse time-frequency representations by Timothy J. Gardner and Marcelo O. Magnasco.
    Code in MATLAB by authors of the paper can be found here:
        https://github.com/earthspecies/spectral_hyperresolution/blob/master/reassignmentgw.m

    Args:
        x (numpy.ndarray of shape (N, C)):
            signal, an array of sampled amplitudes with values on the interval from -1 to 1,
            where N is the number of samples and C number of channels
        q (float):
            the Q of a wavelet
            good values to try when working with tonal sounds are 2, 4, 8 and 1 and 0.5 for impulsive sounds
        tdeci (int): temporal stride in samples, bigger means narrower picture
        over (int):  number of frequencies tried per vertical pixel
        minf (float):  the smallest frequency to visualize
        maxf (float):  the largest frequency to visualize

        natural units: time in samples, frequency [0,1) where 1=sampling rate

        For a more indepth treatment of the parameters please see:
            https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example_in_Python.ipynb
    """
    assert x.ndim == 2, 'signal (x) has to be two dimensional'

    lint = 0.5         # do not follow if reassignment takes you far
    MAXL = 2^27        # maximum length of vector to avoid paging
    eps = 1e-20

    noct = np.array([[noct]])
    over = np.array([[over]])
    tdeci = np.array([[tdeci]])
    minf = np.array([[minf]])
    maxf = np.array([[maxf]])
    N = np.array([[x.shape[0]]])
    xf = fftn(x, axes=[0])

    HT = np.ceil(N/tdeci).astype(np.int)
    HF = np.ceil(-noct*np.log2(minf/maxf)+1).astype(np.int)
    f = np.arange(0, N) / N
    f[f>0.5]=f[f>0.5]-1

    histo = csc_matrix((HT[0][0], HF[0][0]))
    histc = csc_matrix((HT[0][0], HF[0][0]))

    allt = np.zeros((1,0))
    allf = np.zeros((1,0))
    alle = np.zeros((1,0))
    allc = np.zeros((1,0))

    minf = minf.astype(np.float64)
    maxf = maxf.astype(np.float64)
    over = over.astype(np.float64)
    noct = noct.astype(np.float64)

    for log2f0 in np.arange(0, HF.astype(np.float32)*over):
        f0 = minf*2**(log2f0/over/noct)
        sigma = f0/(2*np.pi*q)
        gau = np.exp(-(f-f0)**2 / (2*sigma**2))
        gde = -1/sigma**1 * (f-f0) * gau

        xi = ifftn(gau.T * xf, axes=[0])
        eta = ifftn(gde.T * xf, axes=[0])
        mp = eta / (xi + eps)
        ener = abs(xi)**2

        tins = np.arange(1, N+1).reshape(-1, 1) + np.imag(mp)/(2*np.pi*sigma)
        fins = f0 - np.real(mp)*sigma;

        mask = (abs(mp)<lint) & (fins < maxf) & (fins>minf) & (tins>=1) & (tins<N)
        tins = tins[mask]
        fins = fins[mask]
        ener = ener[mask]

        itins = np.round(tins/tdeci+0.5)
        ifins = np.round(-noct*np.log2(fins/maxf)+1)


        allt = np.hstack((allt, itins))
        allf = np.hstack((allf, ifins))
        alle = np.hstack((alle, ener.reshape(1, -1)))
        allc = np.hstack((allc, 0*itins+1))

        if(len(allt)>MAXL):
            histo = histo + csc_matrix(
                (
                    alle[0],
                    (allt[0].astype(np.int) - 1, allf[0].astype(np.int) - 1)
                ),
                (HT[0][0],HF[0][0])
            )

            histc = histc + csc_matrix(
                (
                    allc[0],
                    (allt[0].astype(np.int) - 1, allf[0].astype(np.int) - 1)
                ),
                (HT[0][0],HF[0][0])
            )
            allt = []
            allf = []
            alle = []
            allc = []

    histo = histo + csc_matrix(
    (
        alle[0],
        (allt[0].astype(np.int) - 1, allf[0].astype(np.int) - 1)
    ),
    (HT[0][0],HF[0][0])
    )

    histc = histc + csc_matrix(
    (
        allc[0],
        (allt[0].astype(np.int) - 1, allf[0].astype(np.int) - 1)
    ),
    (HT[0][0],HF[0][0])
    )

    mm = csc_matrix.max(histc)

    # this operation is very cheap compared to other operations we are performing
    # the warning adds no value nor there exists a good way of addressing it
    with warnings.catch_warnings():
       warnings.simplefilter("ignore")
       histo[histc < np.sqrt(mm)] = 0
    histo.eliminate_zeros()

    return histo
