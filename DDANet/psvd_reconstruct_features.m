function Yhat = psvd_reconstruct_features(X, Vs, use_pinv)
% X    : [Nfeat, Nsamples] real/complex
% Vs   : [Nfeat, L] sparse codebook
% Yhat : [Nfeat, Nsamples]
%
% Reconstruction:
%   Z    = X^T * Vs
%   Xhat = Z * Vs^\dagger
%
% Here we reconstruct in the SAME 832-d implicit feature space.

    arguments
        X {mustBeNumeric}
        Vs {mustBeNumeric}
        use_pinv (1,1) logical = true
    end

    Xrow = X.';   % [Nsamples, Nfeat]

    Z = Xrow * Vs;   % [Nsamples, L]

    if use_pinv
        Vdag = pinv(Vs);         % [L, Nfeat]
    else
        Vdag = Vs';              % only valid if approximately orthonormal
    end

    Xhat_row = Z * Vdag;         % [Nsamples, Nfeat]
    Yhat = Xhat_row.';           % [Nfeat, Nsamples]
    Yhat = real(Yhat);
end