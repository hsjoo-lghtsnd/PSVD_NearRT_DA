function Ht_hat = feature_matrix_to_td_csi(Xhat, Nt, Nr, Ntap)
% feature_matrix_to_td_csi
%
% Input:
%   Xhat   : [Ns, Nt*Nr*Ntap] complex
% Output:
%   Ht_hat : [Ns, Nt, Nr, Ntap] complex

    arguments
        Xhat {mustBeNumeric}
        Nt (1,1) {mustBeInteger, mustBePositive}
        Nr (1,1) {mustBeInteger, mustBePositive}
        Ntap (1,1) {mustBeInteger, mustBePositive}
    end

    Ns = size(Xhat, 1);
    assert(size(Xhat, 2) == Nt * Nr * Ntap, ...
        'Second dimension of Xhat must be Nt*Nr*Ntap.');

    Ht_hat = reshape(Xhat, Ns, Nt, Nr, Ntap);
end