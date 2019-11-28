from scipy.io import loadmat
import sys
import numpy as np
from numpy.fft import *
from scipy.sparse import *
from pylab import figure, cm
from matplotlib.colors import LogNorm
import warnings

def create_reassigned_representation(x, q, tdeci, over, noct, minf, maxf):
    """Create sparse time-frequency representation

    Intended to be a faithful Python implementation of Linear Reassignment as outlined in
        Sparse time-frequency representations by Timothy J. Gardner and Marcelo O. Magnasco.
    Code in Matlab by authors' of the paper can be found here:
        https://github.com/earthspecies/spectral_hyperresolution/blob/master/reassignmentgw.m

    Args:
        x (numpy.ndarray of shape (N, 1)):
            signal, an array of sampled amplitudes with values on the interval from -1 to 1
        q (float):
            the Q of a wavelet
            good values to try when working with tonal sounds are 2, 4, 8 and 1 and 0.5 for impulsive sounds
        tdeci (int): temporal stride in samples, bigger means narrower picture
        over (int):  number of frequencies tried per vertical pixel
        minf (float):  the smallest frequency to visualize
        maxf (float):  the largest frequency to visualize

        natural units: time in samples, frequency [0,1) where 1=sampling rate

        For a more indepth treatment of the paramters please see:
            https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example.ipynb
    """
    assert x.ndim == 2, 'signal (x) has to be two dimensional'
    assert x.shape[1] == 1, 'x.shape has to equal (N, 1)'

    lint = 0.5         # do not follow if reassignment takes you far
    MAXL = 2^27        # maximum length of vector to avoid paging
    eps = 1e-20

    noct = np.array([[noct]])
    over = np.array([[over]])
    tdeci = np.array([[tdeci]])
    minf = np.array([[minf]])
    maxf = np.array([[maxf]])
    N = np.array([[x.size]])
    xf = fftn(x)

    HT = np.ceil(N/tdeci).astype(np.int)
    HF = np.ceil(-noct*np.log2(minf/maxf)+1).astype(np.int)
    f = (np.arange(0, N) / N)

    histo = csc_matrix((HT[0][0], HF[0][0]))
    histc = csc_matrix((HT[0][0], HF[0][0]))

    allt = np.zeros((1,0))
    allf = np.zeros((1,0))
    alle = np.zeros((1,0))
    allc = np.zeros((1,0))

    minf = minf.astype(np.float32)
    maxf = maxf.astype(np.float32)
    over = over.astype(np.float32)
    noct = noct.astype(np.float32)

    for log2f0 in np.arange(0, HF.astype(np.float32)*over):
        f0 = minf*2**(log2f0/over/noct)
        sigma = f0/(2*np.pi*q)
        gau = np.exp(-(f-f0)**2 / (2*sigma**2))
        gde = -1/sigma**1 * (f-f0) * gau

        xi = ifftn(gau.T * xf)
        eta = ifftn(gde.T * xf)
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

    # this operation is very cheap comapared to other operations we are performing
    # the warning adds no value nor there exists a good way of addressing it
    with warnings.catch_warnings():
       warnings.simplefilter("ignore")
       histo[histc < np.sqrt(mm)] = 0
    histo.eliminate_zeros()

    return histo

def plot_spectogram(spectogram, sr, minf, maxf, tdeci):
    '''Plots the spectogram as returned by linear_reassignment.

    sr, minf, maxf, tdeci should have same value as passed to linear_reassignment
    '''
    spectogram_dense = spectogram.todense().T
    minfreq = minf*sr
    maxfreq = maxf*sr

    f = figure(figsize=(6.2,5.6))
    ax = f.add_axes([0.07, 0.02, 0.79, 0.89])
    axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])
    im = ax.matshow(spectogram_dense, cmap=cm.gray_r, norm=LogNorm(vmin=0.01, vmax=np.max(spectogram_dense)))
    t = [0.01, 0.1, 1.0, np.max(spectogram_dense)]
    f.colorbar(im, cax=axcolor, ticks=t, format='$%.2f$')

    ax.set_yticks([spectogram_dense.shape[0]-1, (spectogram_dense.shape[0]-1) // 2, 0])
    midfreq = 2**(np.log2(minfreq) + (np.log2(maxfreq) - np.log2(minfreq)) / 2)
    ax.set_yticklabels([f'{int(f)} Hz' for f in [minfreq, round(midfreq, 0),  maxfreq]])
    ax.set_ylabel('frequency')

    ax.set_xticks([0,  spectogram_dense.shape[1]//2, spectogram_dense.shape[1]])
    ax.set_xticklabels([f'{t}s' for t in [0, round(spectogram_dense.shape[1]/2 * tdeci / sr, 2), round(spectogram_dense.shape[1] * tdeci / sr, 2)]])
    ax.set_xlabel('time')
