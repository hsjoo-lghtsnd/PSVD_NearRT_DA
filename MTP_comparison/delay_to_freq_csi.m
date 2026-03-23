function Hf = delay_to_freq_csi(Ht, Nsub)
% Convert truncated delay-domain CSI back to frequency-domain CSI
% by zero-padding along the tap axis and applying FFT.
%
% Input:
%   Ht   : [Ns, Nt, Nr, Ntap] complex
%   Nsub : target number of subcarriers after zero-padding
%
% Output:
%   Hf   : [Ns, Nt, Nr, Nsub] complex

    arguments
        Ht {mustBeNumeric}
        Nsub (1,1) {mustBeInteger, mustBePositive}
    end

    assert(ndims(Ht) == 4, 'Ht must be a 4-D tensor: [Ns, Nt, Nr, Ntap].');

    [Ns, Nt, Nr, Ntap] = size(Ht);
    assert(Ntap <= Nsub, 'Ntap must satisfy Ntap <= Nsub.');

    % Zero-pad in delay domain
    Hpad = complex(zeros(Ns, Nt, Nr, Nsub));
    Hpad(:,:,:,1:Ntap) = Ht;

    % FFT along delay/tap axis (4th dimension)
    Hf = fft(Hpad, [], 4);
end