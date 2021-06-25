import torch
import math

def complex_multiplication(a, b):
    '''Complex multiplication of a and b

    a and be are tensors of dimensionality [num_channels, n_samples, 2]. The last dimension holds the real part
    as the first entry and complex part as the second.
    '''
    a_real = a[:, :, 0]
    a_imag = a[:, :, 1]
    b_real = b[:, :, 0]
    b_imag = b[:, :, 1]

    ac = a_real * b_real
    bd = a_imag * b_imag
    ad = a_real * b_imag
    bc = a_imag * b_real
    real = ac - bd
    imag = ad + bc

    return torch.cat((real.unsqueeze(-1), imag.unsqueeze(-1)), dim=-1)

def complex_conjugate(t):
    '''
    t is of dimensionality [num_channels, n_samples, 2]
    '''
    t2 = t.clone()
    t2[:, :, 1] *= -1
    return t2

def complex_division(a, b):
    '''divides a by b'''
    numerator = complex_multiplication(a, complex_conjugate(b))
    denominator = b[:, :, 0]**2 + b[:, :, 1]**2
    return numerator / denominator.unsqueeze(-1)

def abs_of_complex_number(t):
    '''
    t is of dimensionality [num_channels, n_samples, 2]
    '''
    t_squared = t**2
    return torch.sqrt(t_squared[:, :, 0] + t_squared[:, :, 1])

def high_resolution_spectrogram(x, q, tdeci, over, noct, minf, maxf, \
        device=torch.device('cpu'), dtype=torch.float32):
    """Create time-frequency representation

    Pytorch implementation of Linear Reassignment as outlined in
        Sparse time-frequency representations by Timothy J. Gardner and Marcelo O. Magnasco.
    Code in Matlab by authors' of the paper can be found here:
        https://github.com/earthspecies/spectral_hyperresolution/blob/master/reassignmentgw.m

    Args:
        x (a tensor or numpy.ndarray of shape (N, C)):
            signal, an array of sampled amplitudes with values on the interval from -1 to 1,
            where N is the number of samples and C number of channels
        q (float):
            the Q of a wavelet
            good values to try when working with tonal sounds are 2, 4, 8 and 1 and 0.5 for impulsive sounds
        tdeci (int): temporal stride in samples, bigger means narrower picture
        over (int):  number of frequencies tried per vertical pixel
        minf (float):  the smallest frequency to visualize
        maxf (float):  the largest frequency to visualize
        device (torch.device): device to run the calculations on
        dtype (torch.dtype):
            torch.float32 or torch.float64, whether to use single or double precision,
            torch.float64 returns the same result as the original MATLAB code
        chunks (int): number of chunks to split the calculation into, useful for managing
            GPU memory footprint, with the default value of 1 the calculation is fully vectorized

        natural units: time in samples, frequency [0,1) where 1=sampling rate

        For a more indepth treatment of the parameters please see:
            https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example_in_Python.ipynb
    """

    assert x.ndim == 2, 'signal (x) has to be two dimensional'
    eps = 1e-20
    lint = 0.5

    x = torch.tensor(x, dtype=dtype, device=device)
    noct = torch.tensor([[noct]], dtype=dtype, device=device)
    over = torch.tensor([[over]], dtype=dtype, device=device)
    tdeci = torch.tensor([[tdeci]], dtype=dtype, device=device)
    minf = torch.tensor([[minf]], dtype=dtype, device=device)
    maxf = torch.tensor([[maxf]], dtype=dtype, device=device)
    N = torch.tensor([[x.shape[0]]], dtype=dtype, device=device)

    xf = torch.fft.fft(x.permute(1,0))
    #interpret result from fft as [real, imag]
    xf = torch.view_as_real(xf)
    HT = torch.ceil(N/tdeci).long()
    HF = torch.ceil(-noct*torch.log2(minf/maxf)+1).long()

    f = (torch.arange(0, N[0][0], device=device) / N)
    f[f>0.5]=f[f>0.5]-1

    histo = torch.zeros(HT[0][0], HF[0][0], device=device, dtype=dtype)
    histc = torch.zeros(HT[0][0], HF[0][0], device=device, dtype=dtype)

    for log2f0 in torch.arange(0, HF[0][0].item()*over.item(), device=device):
        f0 = minf*2**(log2f0/over/noct)
        sigma = f0/(2*math.pi*q)
        gau = torch.exp(-(f-f0)**2 / (2*sigma**2))
        gde = -1/sigma**1 * (f-f0) * gau

        xi = torch.fft.ifft(torch.view_as_complex(gau.T * xf))
        eta = torch.fft.ifft(torch.view_as_complex(gde.T * xf))
        xi = torch.view_as_real(xi)
        eta = torch.view_as_real(eta)
        
        mp = complex_division(eta, xi + eps)
        ener = abs_of_complex_number(xi)**2

        tins = torch.arange(1, (N+1).item(), device=device) +  mp[:, :, 1]/(2*math.pi*sigma)
        fins = f0 - mp[:, :, 0]*sigma
        mask = (abs_of_complex_number(mp)<lint) & (fins < maxf) & (fins>minf) & (tins>=1) & (tins<N)

        tins = tins[mask]
        fins = fins[mask]
        ener = ener[mask]

        itins = torch.round(tins/tdeci+0.5)
        ifins = torch.round(-noct*torch.log2(fins/maxf)+1)
        row_idxs = itins[0].long()-1

        col_idxs = ifins[0].long()-1
        idx_tensor = row_idxs * histo.shape[1] + col_idxs
        histo.put_(idx_tensor, ener, accumulate=True)
        histc.put_(idx_tensor, (0*itins[0]+1), accumulate=True)

    mm = histc.max()
    histo[histc < torch.sqrt(mm)] = 0
    return histo
