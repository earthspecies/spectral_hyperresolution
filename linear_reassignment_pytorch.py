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

    complex_a = torch.cat((real[None, 0], imag[None, 0]), dim=0).permute(1, 0).unsqueeze(0)
    complex_b = torch.cat((real[None, 1], imag[None, 1]), dim=0).permute(1, 0).unsqueeze(0)
    return torch.cat((complex_a, complex_b))

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

def high_resolution_spectogram_pytorch(x, q, tdeci, over, noct, minf, maxf, device='cpu'):

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

    for log2f0 in torch.arange(0, HF[0][0].item()*over.item(), device=device):
        f0 = minf*2**(log2f0/over/noct)
        sigma = f0/(2*math.pi*q)
        gau = torch.exp(-(f-f0)**2 / (2*sigma**2))
        gde = -1/sigma**1 * (f-f0) * gau

        xi = torch.ifft(gau.T * xf, 1)
        eta = torch.ifft(gde.T * xf, 1)
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

def high_resolution_spectogram_pytorch_float32(x, q, tdeci, over, noct, minf, maxf, device='cpu'):

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

        N = torch.DoubleTensor([[x.shape[0]]])
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

    for log2f0 in torch.arange(0, HF[0][0].item()*over.item(), device=device):
        f0 = minf*2**(log2f0/over/noct)
        sigma = f0/(2*math.pi*q)
        gau = torch.exp(-(f-f0)**2 / (2*sigma**2))
        gde = -1/sigma**1 * (f-f0) * gau

        xi = torch.ifft(gau.T * xf, 1)
        eta = torch.ifft(gde.T * xf, 1)
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
