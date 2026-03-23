function Hf_active = delay_to_freq_csi_twosided_fullgrid(Ht, Nfft, K)
% Reconstruct active-band frequency CSI from compact two-sided delay CSI
% using full-grid embedding.
%
% Input:
%   Ht        : [Ns, Nt, Nr, Ntap]
%               = [front taps, back taps]
%   Nfft      : full FFT size used for full-grid reconstruction
%   K         : number of active subcarriers to extract at the center
%
% Output:
%   Hf_active : [Ns, Nt, Nr, K]
%
% Notes:
%   1) Front taps are placed at the beginning of the full delay axis.
%   2) Back taps are placed at the end of the full delay axis.
%   3) FFT is taken on the full grid.
%   4) Only the centered active band is extracted.

    arguments
        Ht {mustBeNumeric}
        Nfft (1,1) {mustBeInteger, mustBePositive}
        K (1,1) {mustBeInteger, mustBePositive}
    end

    assert(ndims(Ht) == 4, 'Ht must be [Ns, Nt, Nr, Ntap].');

    [Ns, Nt, Nr, Ntap] = size(Ht);
    assert(Ntap <= Nfft, 'Ntap must satisfy Ntap <= Nfft.');
    assert(K <= Nfft, 'K must satisfy K <= Nfft.');

    % -----------------------------
    % 1) Expand compact two-sided taps to full delay grid
    % -----------------------------
    nFront = ceil(Ntap/2);
    nBack  = floor(Ntap/2);

    Ht_full = complex(zeros(Ns, Nt, Nr, Nfft));

    % Front taps -> beginning
    Ht_full(:,:,:,1:nFront) = Ht(:,:,:,1:nFront);

    % Back taps -> end
    if nBack > 0
        Ht_full(:,:,:,end-nBack+1:end) = Ht(:,:,:,nFront+1:end);
    end

    % -----------------------------
    % 2) Full-grid FFT
    % -----------------------------
    Hf_full = fft(Ht_full, [], 4);    % [Ns, Nt, Nr, Nfft]

    % -----------------------------
    % 3) Extract centered active band
    % -----------------------------
    nLeft = floor((Nfft - K)/2);
    idxActive = (nLeft + 1):(nLeft + K);

    Hf_active = Hf_full(:,:,:,idxActive);
end