function Ht = freq_to_delay_csi(Hf, Ntap)
% Convert frequency-domain CSI to delay-domain CSI
%
% Input:
%   Hf   : [Ns, Nt, Nr, Nsub] complex
%   Ntap : number of taps to keep
%
% Output:
%   Ht   : [Ns, Nt, Nr, Ntap] complex

    arguments
        Hf {mustBeNumeric}
        Ntap (1,1) {mustBeInteger, mustBePositive}
    end

    assert(ndims(Hf) == 4, 'Hf must be a 4-D tensor: [Ns, Nt, Nr, Nsub].');

    [Ns, Nt, Nr, Nsub] = size(Hf);
    assert(Ntap <= Nsub, 'Ntap must satisfy Ntap <= Nsub.');

    % IFFT along frequency axis (4th dimension)
    HdelayFull = ifft(Hf, [], 4);

    % Keep first Ntap taps
    Ht = HdelayFull(:,:,:,1:Ntap);

    % Explicitly preserve shape
    Ht = reshape(Ht, [Ns, Nt, Nr, Ntap]);
end