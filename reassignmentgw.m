function histocomp = reassignmentgw( x, q, tdeci, over, noct, minf, maxf)
% frequency-line reassignment algorithm for frequency logscale constant Q
% new version of wgreass for whistle tracking. 
% marcellus magnascorum et seanus custode silva  rockefellerensiis 
%       faciebat me anno domine mmxiv - pro bono humanis generi

% you can set GPU to 1 to use parallel
GPU=0;
% natural units: time in samples, frequency [0,1) where 1=sampling rate
% x     signal, 
% q     wavelet's Q (temporal width of wavelet = Q/frequency ) 
%       (>1 follows tone-like sounds, <1 follows click-like sounds)
% tdeci temporal stride in samples, bigger means narrower picture
% over  oversampling (number of frequencies tried per vertical pixel)
% noct  number of divisions per octave (frequency stride in log freq)
% minf  the smallest frequency
% maxf  the largest frequency 
% returns a sparse array with vertical frequency in log units
% shape of the returned array:  twidth= total samples/tdeci (rounded+1)
%                               fwidth= log2(maxf/minf)*noct (rounded+1)

% global parameters
lint = 0.5;       % do not follow if reassignment takes you far
MAXL = 2^27;    % maximum length of vector to avoid paging

tic;
% create state
N=length(x);
% now gpu optional
if(GPU) 
    gx = gpuArray(x);
else
    gx = x;
end

% frequency scan starts by moving to the frequency domain
xf = fft(gx); % (sean) this is an "inverted" spectrogram, calculated from frequencies

HT = ceil(N/tdeci); % (sean) number of final time bins (assuming 'tdeci' is the time window size)
HF = ceil( -noct*log2(minf/maxf)+1); % (sean) number of final freq bins, i.e. number of octaves * divisions between fmin and fmax. +1 so that maxf is indexed over in log2f0, below
histo = spalloc(HT,HF,N);
histc = spalloc(HT,HF,N); 

% prepare the masks

if(GPU) f = gpuArray( (0:N-1)/N ); else f=(0.0:N-1)/N; end

f(f>0.5)=f(f>0.5)-1; % (sean) f: (-0.5, 0.5), N increments (maximum possibe frequency divisions)

allt = [];
allf = [];
alle = [];
allc = [];


for log2f0=0:( (HF*over) -1) % (sean) indices of oversampled (sub-subdivided) subdivided octaves
    f0 = minf*2^(log2f0/over/noct); % (sean) frequencies (Hz) corresponding to octave subdivision indices
    sigma = f0/(2*pi*q); % changed to 1/2pi % (sean) Gaussian parameter: increases with center frequency (f0), decreases with q (set bigger for more freq resolution)
    gau = exp( -(f-f0).^2 / (2*sigma^2)); % (sean) Gaussian window over entire frequency axis (centered at f0)
    gde = -1/sigma^1 * (f-f0) .* gau; % (sean) "eta" window over entire frequency axis (centered at f0)
    
    % compute reassignment operator
    xi = ifft( gau' .* xf); % (sean) calculate ifft of Gaussian-windowed snippet 
    eta= ifft( gde' .* xf); % (sean) calculate ifft of eta-windowed snippet
    mp = ( eta ./ xi ); % (sean) calculate complex shift (again, complex conjugate operation left out...real signal thing?)
    ener = abs(xi).^2; % (sean) calculate ifft energy
    
    % compute instantaneous time and frequency
    tins = (1:N)' + imag(mp)/(2*pi*sigma); % (sean) *why is this positive? Due to inverse operation? Incidentally, for some reason Tim seems to divide both time and freq shifts (mp) by 2pi
    fins =     f0 - real(mp)*sigma;
    % mask the results to the histogram domain
    mask = (abs(mp)<lint) & (fins < maxf) & (fins>minf) & (tins>=1) & (tins<N); % (sean) note that f0 > maxf possible
    tins = tins( mask );
    fins = fins( mask );
    ener = ener( mask );
    % histogram!
    itins = gather( round(tins/tdeci+0.5)); % (sean) bin times... +0.5 ensures bin indices > 0
    ifins = gather( round( - noct*log2(fins/maxf)+1) ); % (sean) bin freqs... +1 ensures bin indices > 0 (why is +1 used...?)

    allt = [ allt; itins ];
    allf = [ allf; ifins ];
    alle = [ alle; gather( ener ) ];
    allc = [ allc; (0*itins+1) ];
    if(length(allt)>MAXL)
        disp(['fft    took ' num2str([ toc f0 (log2f0/HF/over)])] );
        tic;
        histo=histo+sparse(allt,allf,alle,HT,HF);
        histc=histc+sparse(allt,allf,allc,HT,HF);
        allt = [];
        allf = [];
        alle = [];
        allc = []; 
        disp(['sparse took ' num2str([ toc f0 (log2f0/HF/over)])] );
        tic;
    end

end

histo=histo+sparse(allt,allf,alle,HT,HF);
histc=histc+sparse(allt,allf,allc,HT,HF);
mm = max(max(histc)); 
histo(histc<sqrt(mm)) = 0;
histocomp=spalloc(HT,HF,sum(sum(histo>0))); 
histocomp(histo>0)=histo(histo>0); 
disp([ 'done_reassignment in ', num2str(toc), ' s'])
return