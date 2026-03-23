function Ht = freq_to_delay_csi_twosided_fullgrid(Hf_active, Nfft, Ntap)
% Convert active-band frequency CSI to two-sided truncated delay CSI
% using full-grid zero-padding (guard-band style).
%
% Input:
%   Hf_active : [Ns, Nt, Nr, K]
%   Nfft      : full FFT size (>= K)
%   Ntap      : number of kept taps in compact two-sided form
%
% Output:
%   Ht        : [Ns, Nt, Nr, Ntap]
%               = [front taps, back taps]
%
% Notes:
%   1) Active subcarriers are centered in the full FFT grid.
%   2) Zero-padding is applied on both sides (guard-band style).
%   3) IFFT is taken on the full grid.
%   4) Delay taps are compacted by keeping both front and back taps.

    arguments
        Hf_active {mustBeNumeric}
        Nfft (1,1) {mustBeInteger, mustBePositive}
        Ntap (1,1) {mustBeInteger, mustBePositive}
    end

    assert(ndims(Hf_active) == 4, 'Hf_active must be [Ns, Nt, Nr, K].');

    [Ns, Nt, Nr, K] = size(Hf_active);
    assert(Nfft >= K, 'Nfft must satisfy Nfft >= K.');
    assert(Ntap <= Nfft, 'Ntap must satisfy Ntap <= Nfft.');

    % -----------------------------
    % 1) Expand active band to full grid (centered)
    % -----------------------------
    Hf_full = complex(zeros(Ns, Nt, Nr, Nfft));

    nLeft = floor((Nfft - K)/2);
    idxActive = (nLeft + 1):(nLeft + K);

    Hf_full(:,:,:,idxActive) = Hf_active;

    % -----------------------------
    % 2) Full-grid IFFT
    % -----------------------------
    Ht_full = ifft(Hf_full, [], 4);   % [Ns, Nt, Nr, Nfft]

    % -----------------------------
    % 3) Two-sided compact truncation
    % -----------------------------
    nFront = ceil(Ntap/2);
    nBack  = floor(Ntap/2);

    Hfront = Ht_full(:,:,:,1:nFront);

    if nBack > 0
        Hback = Ht_full(:,:,:,end-nBack+1:end);
        Ht = cat(4, Hfront, Hback);
    else
        Ht = Hfront;
    end
end