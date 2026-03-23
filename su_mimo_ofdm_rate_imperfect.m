function [R, gamma, Pcell, Wcell] = su_mimo_ofdm_rate_imperfect(Horg, H, Htilde, Nsym, Ptot, N0, opts)
%SU_MIMO_OFDM_RATE_IMPERFECT
% Compute achievable spectral efficiency R for SU-MIMO-OFDM under
% - true channel:     Horg^F
% - imperfect CSIR:   H^F = Horg^F + N^F  (used for ZF combining)
% - imperfect CSIT:   Htilde^F            (used for SVD precoding)
%
% Inputs (required)
%   Horg    : [Nt, Nr, Nsub] complex, true channel per subcarrier
%   H       : [Nt, Nr, Nsub] complex, noisy CSIR per subcarrier
%   Htilde  : [Nt, Nr, Nsub] complex, reconstructed CSIT per subcarrier
%   Nsym    : number of spatial layers (<= min(Nt,Nr))
%   Ptot    : total transmit power across all Tx antennas
%   N0      : noise variance per receive antenna (n ~ CN(0, N0 I))
%
% opts (optional struct)
%   opts.use_pinv        : true/false (default: true)  % robust ZF
%   opts.reg_epsilon     : >=0 (default: 0)            % if >0, uses RZF-style regularization
%   opts.return_cells    : true/false (default: false) % return Pcell/Wcell if true
%
% Outputs
%   R       : scalar, achievable spectral efficiency averaged over subcarriers
%   gamma   : [Nsym, Nsub] realized post-precoding SINR per layer/subcarrier
%   Pcell   : 1xNsub cell of precoders (optional)
%   Wcell   : 1xNsub cell of combiners (optional)

    arguments
        Horg {mustBeNumeric}
        H {mustBeNumeric}
        Htilde {mustBeNumeric}
        Nsym (1,1) {mustBeInteger, mustBePositive}
        Ptot (1,1) double {mustBePositive}
        N0 (1,1) double {mustBePositive}
        opts struct
    end

    if ~isfield(opts, 'use_pinv'),     opts.use_pinv = true; end
    if ~isfield(opts, 'reg_epsilon'),  opts.reg_epsilon = 0; end
    if ~isfield(opts, 'return_cells'), opts.return_cells = false; end

    % ---- Dimension handling: inputs are [Nt, Nr, Nsub], internally use [Nr, Nt, Nsub]
    % so that Hk is Nr x Nt consistent with paper.
    [Nt, Nr, Nsub] = size(Horg);
    assert(all(size(H) == [Nt, Nr, Nsub]) && all(size(Htilde) == [Nt, Nr, Nsub]), ...
        'Horg, H, Htilde must have the same size [Nt, Nr, Nsub].');
    assert(Nsym <= min(Nt, Nr), 'Nsym must be <= min(Nt, Nr).');

    Horg_rtn = permute(Horg,  [2 1 3]);   % [Nr, Nt, Nsub]
    H_rtn    = permute(H,     [2 1 3]);   % [Nr, Nt, Nsub]
    Ht_rtn   = permute(Htilde,[2 1 3]);   % [Nr, Nt, Nsub]

    eta = Ptot / (Nsub * Nsym);

    gamma = zeros(Nsym, Nsub);
    if opts.return_cells
        Pcell = cell(1, Nsub);
        Wcell = cell(1, Nsub);
    else
        Pcell = [];
        Wcell = [];
    end

    for k = 1:Nsub
        Horg_k = Horg_rtn(:,:,k);  % Nr x Nt
        H_k    = H_rtn(:,:,k);     % Nr x Nt (noisy CSIR)
        Ht_k   = Ht_rtn(:,:,k);    % Nr x Nt (reconstructed CSIT)

        % ---- Precoder P[k] from truncated SVD of Htilde_k
        % Ht_k = U * S * V^H, take V(:,1:Nsym)
        [~, ~, V] = svd(Ht_k, 'econ');      % V: Nt x r
        Pk = V(:, 1:Nsym);                  % Nt x Nsym

        % Normalize ||Pk||_F^2 = Nsym  (usually already true if columns orthonormal)
        fro2 = sum(abs(Pk(:)).^2);
        if fro2 > 0
            Pk = Pk * sqrt(Nsym / fro2);
        end

        % ---- ZF combiner W[k] based on noisy CSIR H_k
        % Effective channel: Heff = H_k * Pk  (Nr x Nsym)
        Heff = H_k * Pk;

        % Compute Wk = Heff * (Heff^H Heff)^{-1}   (Nr x Nsym)
        G = (Heff' * Heff);  % Nsym x Nsym (Hermitian)
        if opts.reg_epsilon > 0
            G = G + opts.reg_epsilon * eye(Nsym);
        end

        if opts.use_pinv
            Wk = Heff * pinv(G);
        else
            % may fail if ill-conditioned; left here for speed when well-conditioned
            Wk = Heff / G;
        end

        % ---- Effective scalar channel matrix on TRUE channel:
        % g_{l,m}[k] = w_l^H * Horg_k * p_m
        % Equivalent: Gtrue = Wk^H * Horg_k * Pk  (Nsym x Nsym)
        Gtrue = (Wk' * Horg_k * Pk);

        % ---- SINR per layer
        diagPow = abs(diag(Gtrue)).^2;                       % |g_ll|^2
        totalRowPow = sum(abs(Gtrue).^2, 2);                 % sum_m |g_lm|^2
        interfPow = totalRowPow - diagPow;                   % sum_{m!=l} |g_lm|^2

        wNorm2 = sum(abs(Wk).^2, 1).';                       % ||w_l||^2, (Nsym x 1)

        gamma(:, k) = (eta * diagPow) ./ (eta * interfPow + N0 * wNorm2);

        if opts.return_cells
            Pcell{k} = Pk;
            Wcell{k} = Wk;
        end
    end

    % ---- Spectral efficiency averaged over subcarriers
    R = mean(sum(log2(1 + gamma), 1));  % (1/Nsub) * sum_k sum_l log2(1+gamma_l[k])
end