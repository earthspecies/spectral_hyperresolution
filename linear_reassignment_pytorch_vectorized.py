import torch
import math

def complex_division(a, b):
    '''divides a by b'''
    numerator = complex_multiplication(a, complex_conjugate(b))
    denominator = b[:, :, 0]**2 + b[:, :, 1]**2
    return numerator / denominator.unsqueeze(-1)

def complex_divisions(a, b):
    '''divides a by b'''
    numerator = complex_multiplications(a, complex_conjugates(b))
    denominator = b[:, :, :, 0]**2 + b[:, :, :, 1]**2
    return numerator / denominator.unsqueeze(-1)


def complex_conjugates(t):
    '''
    t is of dimensionality [num_channels, n_samples, 2]
    '''
    t2 = t.clone()
    t2[:, :, :, 1] *= -1
    return t2

def complex_multiplications(a, b):
    '''Complex multiplication of a and b

    a and be are tensors of dimensionality [num_channels, log2f0s.shape[0], n_samples, 2]. The last dimension holds the real part
    as the first entry and complex part as the second.
    '''
    a_real = a[:, :, :, 0]
    a_imag = a[:, :, :, 1]
    b_real = b[:, :, :, 0]
    b_imag = b[:, :, :, 1]

    ac = a_real * b_real
    bd = a_imag * b_imag
    ad = a_real * b_imag
    bc = a_imag * b_real
    real = ac - bd
    imag = ad + bc

    return torch.cat((real.unsqueeze(-1),imag.unsqueeze(-1)), dim=-1)

def abs_of_complex_numbers(t):
    '''
    t is of dimensionality [num_channels, log2f0s.shape[0], n_samples, 2]
    '''
    t_squared = t**2
    return torch.sqrt(t_squared[:, :, :, 0] + t_squared[:, :, :, 1])

def high_resolution_spectogram(x, q, tdeci, over, noct, minf, maxf, device='cpu'):
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

        natural units: time in samples, frequency [0,1) where 1=sampling rate

        For a more indepth treatment of the parameters please see:
            https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example_in_Python.ipynb
    """

    assert x.ndim == 2, 'signal (x) has to be two dimensional'
    eps = 1e-20
    lint = 0.5

    if device == 'cpu':
        x = torch.DoubleTensor(x)
        noct = torch.DoubleTensor([[noct]])
        over = torch.DoubleTensor([[over]])
        tdeci = torch.DoubleTensor([[tdeci]])
        minf = torch.DoubleTensor([[minf]])
        maxf = torch.DoubleTensor([[maxf]])

        N = torch.DoubleTensor([[x.shape[0]]])
    elif device == 'cuda':
        x = torch.cuda.DoubleTensor(x)
        noct = torch.cuda.DoubleTensor([[noct]])
        over = torch.cuda.DoubleTensor([[over]])
        tdeci = torch.cuda.DoubleTensor([[tdeci]])
        minf = torch.cuda.DoubleTensor([[minf]])
        maxf = torch.cuda.DoubleTensor([[maxf]])
        N = torch.cuda.DoubleTensor([[x.shape[0]]])

    xf = torch.rfft(x.permute(1,0), 1, onesided=False)

    HT = torch.ceil(N/tdeci).long()
    HF = torch.ceil(-noct*torch.log2(minf/maxf)+1).long()

    f = (torch.arange(0, N[0][0], device=device) / N)
    f[f>0.5]=f[f>0.5]-1

    histo = torch.zeros(HT[0][0], HF[0][0], device=device, dtype=torch.float64)
    histc = torch.zeros(HT[0][0], HF[0][0], device=device, dtype=torch.float64)

    log2f0s = torch.arange(0, HF[0][0].item()*over.item(), device=device)
    xfs = xf.unsqueeze(1).expand(xf.shape[0], log2f0s.shape[0], xf.shape[1], xf.shape[2])
    f0s = minf*2**(log2f0s/over/noct).unsqueeze(-1).expand(1, log2f0s.shape[0], f.shape[1])
    sigmas = f0s/(2*math.pi*q)
    fs = f[:, None].expand(1, log2f0s.shape[0], f.shape[1])
    gaus = torch.exp(-(fs-f0s)**2 / (2*sigmas**2))
    gdes = -1/sigmas**1 * (fs-f0s) * gaus

    xis = torch.ifft(gaus.unsqueeze(-1) * xfs, 1)
    etas = torch.ifft(gdes.unsqueeze(-1) * xfs, 1)
    mps = complex_divisions(etas, xis + eps)
    eners = abs_of_complex_numbers(xis)**2

    tins = torch.arange(1, (N+1).item(), device=device) +  (mps[:, :, :, 1]/(2*math.pi*sigmas))
    fins = f0s - mps[:, :, :, 0]*sigmas
    mask = (abs_of_complex_numbers(mps)<lint) & (fins < maxf) & (fins>minf) & (tins>=1) & (tins<N)
    tins = tins[mask]
    fins = fins[mask]
    eners = eners[mask]

    itins = torch.round(tins/tdeci+0.5)
    ifins = torch.round(-noct*torch.log2(fins/maxf)+1)
    row_idxs = itins[0].long()-1

    col_idxs = ifins[0].long()-1
    idx_tensor = row_idxs * histo.shape[1] + col_idxs

    histo.put_(idx_tensor, eners, accumulate=True)
    histc.put_(idx_tensor, (0*itins[0]+1), accumulate=True)

    mm = histc.max()
    histo[histc < torch.sqrt(mm)] = 0
    return histo

def high_resolution_spectogram_float32(x, q, tdeci, over, noct, minf, maxf, device='cpu'):
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

        natural units: time in samples, frequency [0,1) where 1=sampling rate

        For a more indepth treatment of the parameters please see:
            https://github.com/earthspecies/spectral_hyperresolution/blob/master/linear_reassignment_example_in_Python.ipynb
    """

    assert x.ndim == 2, 'signal (x) has to be two dimensional'
    eps = 1e-20
    lint = 0.5

    if device == 'cpu':
        x = torch.FloatTensor(x)
        noct = torch.FloatTensor([[noct]])
        over = torch.FloatTensor([[over]])
        tdeci = torch.FloatTensor([[tdeci]])
        minf = torch.FloatTensor([[minf]])
        maxf = torch.FloatTensor([[maxf]])
        N = torch.FloatTensor([[x.shape[0]]])
    elif device == 'cuda':
        x = torch.cuda.FloatTensor(x)
        noct = torch.cuda.FloatTensor([[noct]])
        over = torch.cuda.FloatTensor([[over]])
        tdeci = torch.cuda.FloatTensor([[tdeci]])
        minf = torch.cuda.FloatTensor([[minf]])
        maxf = torch.cuda.FloatTensor([[maxf]])
        N = torch.cuda.FloatTensor([[x.shape[0]]])

    xf = torch.rfft(x.permute(1,0), 1, onesided=False)

    HT = torch.ceil(N/tdeci).long()
    HF = torch.ceil(-noct*torch.log2(minf/maxf)+1).long()

    f = (torch.arange(0, N[0][0], device=device) / N)
    f[f>0.5]=f[f>0.5]-1

    histo = torch.zeros(HT[0][0], HF[0][0], device=device, dtype=torch.float32)
    histc = torch.zeros(HT[0][0], HF[0][0], device=device, dtype=torch.float32)

    log2f0s = torch.arange(0, HF[0][0].item()*over.item(), device=device)
    xfs = xf.unsqueeze(1).expand(xf.shape[0], log2f0s.shape[0], xf.shape[1], xf.shape[2])
    f0s = minf*2**(log2f0s/over/noct).unsqueeze(-1).expand(1, log2f0s.shape[0], f.shape[1])
    sigmas = f0s/(2*math.pi*q)
    fs = f[:, None].expand(1, log2f0s.shape[0], f.shape[1])
    gaus = torch.exp(-(fs-f0s)**2 / (2*sigmas**2))
    gdes = -1/sigmas**1 * (fs-f0s) * gaus

    xis = torch.ifft(gaus.unsqueeze(-1) * xfs, 1)
    etas = torch.ifft(gdes.unsqueeze(-1) * xfs, 1)
    mps = complex_divisions(etas, xis + eps)
    eners = abs_of_complex_numbers(xis)**2

    tins = torch.arange(1, (N+1).item(), device=device) +  (mps[:, :, :, 1]/(2*math.pi*sigmas))
    fins = f0s - mps[:, :, :, 0]*sigmas
    mask = (abs_of_complex_numbers(mps)<lint) & (fins < maxf) & (fins>minf) & (tins>=1) & (tins<N)
    tins = tins[mask]
    fins = fins[mask]
    eners = eners[mask]

    itins = torch.round(tins/tdeci+0.5)
    ifins = torch.round(-noct*torch.log2(fins/maxf)+1)
    row_idxs = itins[0].long()-1

    col_idxs = ifins[0].long()-1
    idx_tensor = row_idxs * histo.shape[1] + col_idxs

    histo.put_(idx_tensor, eners, accumulate=True)
    histc.put_(idx_tensor, (0*itins[0]+1), accumulate=True)

    mm = histc.max()
    histo[histc < torch.sqrt(mm)] = 0
    return histo
