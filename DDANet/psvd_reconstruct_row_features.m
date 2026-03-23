function Xhat = psvd_reconstruct_row_features(X, Vs, use_pinv)
% psvd_reconstruct_row_features
%
% Input:
%   X    : [Nsamples, Nfeat] real or complex
%   Vs   : [Nfeat, L] codebook
% Output:
%   Xhat : [Nsamples, Nfeat]
%
% Reconstruction:
%   Z    = X * Vs
%   Xhat = Z * Vs^\dagger
%
% Unlike the implicit-feature version, this function does NOT force
% the output back to real, because time-domain CSI is complex-valued.

    arguments
        X {mustBeNumeric}
        Vs {mustBeNumeric}
        use_pinv (1,1) logical = true
    end

    Z = X * Vs;   % [Nsamples, L]

    if use_pinv
        Vdag = pinv(Vs);   % [L, Nfeat]
    else
        Vdag = Vs';
    end

    Xhat = Z * Vdag;      % [Nsamples, Nfeat]
end