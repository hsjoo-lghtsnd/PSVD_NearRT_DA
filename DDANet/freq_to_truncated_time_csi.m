function Ht = freq_to_truncated_time_csi(Hf, Ntap)
% freq_to_truncated_time_csi
%
% Input:
%   Hf   : [Ns, Nt, Nr, Nsub] complex
% Output:
%   Ht   : [Ns, Nt, Nr, Ntap] complex
%
% Apply IDFT along subcarrier dimension and keep first Ntap taps.

    arguments
        Hf {mustBeNumeric}
        Ntap (1,1) {mustBeInteger, mustBePositive}
    end

    [Ns, Nt, Nr, Nsub] = size(Hf);
    assert(Ntap <= Nsub, 'Ntap must be <= Nsub.');

    Ht_full = ifft(Hf, [], 4);   % IDFT on subcarrier axis
    Ht = Ht_full(:,:,:,1:Ntap);
end