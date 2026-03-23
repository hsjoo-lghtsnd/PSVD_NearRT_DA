function X = td_csi_to_feature_matrix(Ht)
% td_csi_to_feature_matrix
%
% Input:
%   Ht : [Ns, Nt, Nr, Ntap] complex
% Output:
%   X  : [Ns, Nt*Nr*Ntap] complex
%
% Each row is one vectorized time-domain CSI sample.

    arguments
        Ht {mustBeNumeric}
    end

    [Ns, Nt, Nr, Ntap] = size(Ht);
    X = reshape(Ht, Ns, Nt * Nr * Ntap);
end