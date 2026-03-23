function [Vs, sigmaL, V, info] = psvd_codebook_real(Htrain, kappa0, rho0, T)
%PSVD_CODEBOOK Fast sparse codebook construction via PSVD
%
% Inputs
%   Htrain : [Ntrain x Nfeat] complex training matrix
%   kappa0 : maximum compression ratio parameter
%   rho0   : target sparsity ratio in [0, 1]
%            rho0 = fraction to be zeroed
%   T      : number of orthogonal-iteration steps
%
% Outputs
%   Vs     : [Nfeat x L] sparse codebook
%   sigmaL : [L x 1] dominant singular values
%   V      : [Nfeat x L] dense reconstructed right singular vectors
%   info   : struct with auxiliary information
%
% Algorithm
%   1) L = min(floor(Nfeat*kappa0), Ntrain)
%   2) M = floor((1-rho0)*Nfeat)
%   3) G = Htrain*Htrain'
%   4) Orthogonal iteration on G
%   5) Rayleigh quotient sorting
%   6) Reconstruct dense right singular vectors
%   7) Keep only M largest-magnitude entries per column
%
% Notes
%   - Assumes Ntrain << Nfeat, as in the paper.
%   - If sigma_i is numerically zero, the corresponding column of V is set to zero.
%   - qr(X,0) is used for thin QR.

    % ---------------------------
    % Input checks
    % ---------------------------
    if nargin < 4
        error('Usage: [Vs, sigmaL, V, info] = psvd_codebook(Htrain, kappa0, rho0, T)');
    end

    [Ntrain, Nfeat] = size(Htrain);

    if ~isscalar(kappa0) || ~isreal(kappa0) || kappa0 <= 0
        error('kappa0 must be a positive real scalar.');
    end
    if ~isscalar(rho0) || ~isreal(rho0) || rho0 < 0 || rho0 > 1
        error('rho0 must be a real scalar in [0, 1].');
    end
    if ~isscalar(T) || T < 0 || floor(T) ~= T
        error('T must be a nonnegative integer.');
    end

    % ---------------------------
    % Step 1: rank / codeword count
    % ---------------------------
    L = min(floor(Nfeat * kappa0), Ntrain);
    if L < 1
        error('Computed L is zero. Increase kappa0 or check Nfeat.');
    end

    % ---------------------------
    % Step 2: number of kept entries
    % ---------------------------
    M = floor((1 - rho0) * Nfeat);
    M = max(0, min(M, Nfeat));

    % ---------------------------
    % Step 3: Gram matrix
    % ---------------------------
    G = Htrain * Htrain';   % [Ntrain x Ntrain], Hermitian PSD

    % ---------------------------
    % Step 4: orthogonal iteration
    % ---------------------------
    % Q = randn(Ntrain, L) + 1j * randn(Ntrain, L);
    Q = randn(Ntrain, L);
    [Q, ~] = qr(Q, 0);

    for t = 1:T
        Y = G * Q;
        [Q, ~] = qr(Y, 0);
    end

    % ---------------------------
    % Step 5: Rayleigh quotients
    % lambda_i = q_i^H G q_i
    % ---------------------------
    lambda = zeros(L, 1);
    for i = 1:L
        qi = Q(:, i);
        lambda(i) = real(qi' * G * qi);
    end

    [lambdaSorted, sortIdx] = sort(lambda, 'descend');
    Q = Q(:, sortIdx);

    % ---------------------------
    % Step 6: singular values
    % sigma_i = sqrt(max(lambda_i, 0))
    % ---------------------------
    sigmaL = sqrt(max(lambdaSorted, 0));

    % ---------------------------
    % Step 7: reconstruct dense right singular vectors
    % V = Htrain^H * Q * diag(sigmaL)^(-1)
    % ---------------------------
    V = zeros(Nfeat, L, 'like', Htrain);
    tolSigma = 1e-12;

    HQ = Htrain' * Q;   % [Nfeat x L]

    for i = 1:L
        if sigmaL(i) > tolSigma
            V(:, i) = HQ(:, i) / sigmaL(i);
        else
            V(:, i) = 0;
        end
    end

    % ---------------------------
    % Step 8: zeroing each column
    % keep M largest-magnitude entries
    % ---------------------------
    Vs = zeros(Nfeat, L, 'like', Htrain);

    if M == 0
        % keep nothing
        Vs(:) = 0;

    elseif M >= Nfeat
        % keep all
        Vs = V;

    else
        for i = 1:L
            vi = V(:, i);

            % indices of M largest magnitudes
            [~, idx] = maxk(abs(vi), M);

            vsi = zeros(Nfeat, 1, 'like', Htrain);
            vsi(idx) = vi(idx);

            Vs(:, i) = vsi;
        end
    end

    % ---------------------------
    % Auxiliary outputs
    % ---------------------------
    info = struct();
    info.Ntrain = Ntrain;
    info.Nfeat  = Nfeat;
    info.L      = L;
    info.M      = M;
    info.lambda = lambdaSorted;
    info.sortIdx = sortIdx;
    info.rho0   = rho0;
    info.kappa0 = kappa0;
    info.T      = T;
end