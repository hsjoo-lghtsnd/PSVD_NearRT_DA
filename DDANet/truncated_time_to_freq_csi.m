function Hf_hat = truncated_time_to_freq_csi(Ht, Nsub)
% truncated_time_to_freq_csi
%
% Input:
%   Ht     : [Ns, Nt, Nr, Ntap] complex
% Output:
%   Hf_hat : [Ns, Nt, Nr, Nsub] complex
%
% Zero-pad from Ntap to Nsub and apply FFT along delay/subcarrier axis.

    arguments
        Ht {mustBeNumeric}
        Nsub (1,1) {mustBeInteger, mustBePositive}
    end

    [Ns, Nt, Nr, Ntap] = size(Ht);
    assert(Ntap <= Nsub, 'Ntap must be <= Nsub.');

    Hpad = complex(zeros(Ns, Nt, Nr, Nsub, 'like', Ht));
    Hpad(:,:,:,1:Ntap) = Ht;

    Hf_hat = fft(Hpad, [], 4);
end