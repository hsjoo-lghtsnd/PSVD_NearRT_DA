function out = psvd_codebook_flop(H_train, kappa0, rho0, T, opts)
% psvd_codebook_flop
% PSVD: Fast Codebook Construction (per Algorithm 2)
%
% INPUTS:
%   H_train : complex matrix, size (N_train x N_feat)
%   kappa0  : max CR, sets L = min(floor(kappa0*N_feat), N_train)
%   rho0    : target sparsity in [0,1).  Zeroing ratio per column.
%             keep count M = floor((1-rho0)*N_feat).
%   T       : orthogonal-iteration count (fixed, for explicit counting)
%   opts    : (optional) struct with fields:
%       .cmp_model : 'select' (default) or 'sort'
%                    - 'select': comparisons ~= c_sel * N_feat per column (expected)
%                    - 'sort'  : comparisons ~= N_feat*ceil(log2(N_feat)) per column (rough upper)
%       .c_sel     : constant for 'select' comparisons model (default 3)
%       .do_ritz   : true/false, add small LxL Ritz refinement (default false)
%
% OUTPUT (struct out):
%   out.Vs            : sparse codebook (N_feat x L_eff)
%   out.sigma         : singular values (L_eff x 1)
%   out.V_dense       : dense V (N_feat x L_eff) before thresholding
%   out.UL            : left vectors (N_train x L_eff)
%   out.G             : Gram matrix (N_train x N_train)
%   out.flops         : step-wise real-FLOP counts (arithmetic only)
%   out.comparisons   : comparison-count model for Step D
%   out.info          : metadata (N_train, N_feat, L, M, T, etc.)
%
% FLOP model (arithmetic only):
%   complex multiply = 6 real FLOPs
%   complex add      = 2 real FLOPs
%   real scalar * complex = 2 real FLOPs
%   real sqrt        = 1 FLOP (bookkeeping)
%   real division    = 1 FLOP (bookkeeping)
%
% Notes:
%   - Step D (hard thresholding) is comparison-dominated; we report comparisons separately.
%   - Sorting / selection costs are NOT included in FLOPs.

    if nargin < 5, opts = struct(); end
    if ~isfield(opts,'cmp_model'), opts.cmp_model = 'select'; end
    if ~isfield(opts,'c_sel'), opts.c_sel = 3; end
    if ~isfield(opts,'do_ritz'), opts.do_ritz = false; end

    [m, n] = size(H_train);

    % Validate
    if ~isnumeric(H_train) || ndims(H_train) ~= 2
        error('H_train must be a 2-D numeric matrix.');
    end
    if ~isscalar(kappa0) || kappa0 <= 0
        error('gamma0 must be a positive scalar.');
    end
    if ~isscalar(rho0) || rho0 < 0 || rho0 >= 1
        error('rho0 must be in [0,1). (target sparsity / zeroing ratio)');
    end
    if ~isscalar(T) || T < 0 || T ~= floor(T)
        error('T must be a nonnegative integer.');
    end

    % Algorithm spec
    L = min(floor(kappa0 * n), m);
    L = max(L, 1);

    % keep count M = floor((1-rho0)*N_feat)
    M = floor((1 - rho0) * n);
    M = max(0, min(M, n));

    %% ---------------------------------------------------------------
    % Step A: Gram matrix G = H*H^H
    %% ---------------------------------------------------------------
    G = H_train * H_train';   % m x m

    % FLOP count: exploit Hermitian (conceptually upper triangle only)
    cmul_A = n * m * (m + 1) / 2;
    cadd_A = (n - 1) * m * (m + 1) / 2;
    flop_A = 6 * cmul_A + 2 * cadd_A;   % = (4n-1)m(m+1)

    %% ---------------------------------------------------------------
    % Step B: fixed-T orthogonal iteration on G with explicit MGS QR
    %% ---------------------------------------------------------------
    Q0 = randn(m, L) + 1j * randn(m, L);
    [Q, flop_QR_init, detail_QR_init] = mgs_qr_count(Q0);

    % Per-iteration matmul: G*Q (m x m)*(m x L)
    cmul_Bmm = m * m * L;
    cadd_Bmm = m * (m - 1) * L;
    flop_B_matmul_each = 6 * cmul_Bmm + 2 * cadd_Bmm;  % = 8m^2L - 2mL

    flop_B_matmul_total = 0;
    flop_B_qr_total = flop_QR_init;

    for t = 1:T
        Y = G * Q;
        flop_B_matmul_total = flop_B_matmul_total + flop_B_matmul_each;

        [Q, flop_qr_t] = mgs_qr_count(Y);
        flop_B_qr_total = flop_B_qr_total + flop_qr_t;
    end

    % Final GQ for Rayleigh and (optional) Ritz
    Z = G * Q;
    flop_B_final_GQ = flop_B_matmul_each;

    % Rayleigh quotients: lambda_i = q_i^H z_i
    lambda = zeros(L,1);
    flop_B_rayleigh = 0;
    for i = 1:L
        lambda(i) = real(Q(:,i)' * Z(:,i));
        flop_B_rayleigh = flop_B_rayleigh + (8*m - 2); % one complex inner product
    end

    % Optional: small Ritz refinement to reduce mode mixing
    flop_B_ritz = 0;
    if opts.do_ritz
        % B = Q^H G Q = Q^H Z  (L x L)
        % Matrix multiply (L x m)*(m x L):
        % cmul = L*m*L, cadd = L*(m-1)*L
        cmul_Britz = L * m * L;
        cadd_Britz = L * (m - 1) * L;
        flop_B_ritz = 6*cmul_Britz + 2*cadd_Britz;

        Bsmall = Q' * Z;
        Bsmall = (Bsmall + Bsmall')/2;
        [W, D] = eig(Bsmall, 'vector');  %#ok<ASGLU> (eig FLOPs not counted: impl-dependent)
        [lambdaR, idxR] = sort(real(D), 'descend');
        W = W(:, idxR);
        Q = Q * W;
        lambda = lambdaR;
        % (sorting comparisons not counted)
    else
        [lambda, idx] = sort(lambda, 'descend');
        Q = Q(:, idx);
    end

    lambda(lambda < 0) = 0;
    sL = sqrt(lambda);  % sigma_i = sqrt(lambda_i)

    % Prune tiny singular values (numerical)
    tol = max(size(H_train)) * eps(max(sL + 1));
    keep_mode = sL > tol;
    Q  = Q(:, keep_mode);
    sL = sL(keep_mode);
    L_eff = numel(sL);

    UL = Q;

    %% ---------------------------------------------------------------
    % Step C: V = H^H * U_L * diag(sigma)^(-1)
    %% ---------------------------------------------------------------
    if L_eff == 0
        V_dense = zeros(n, 0, class(H_train));
        flop_C1 = 0; flop_C2 = 0; flop_C = 0;
        cmul_C1 = 0; cadd_C1 = 0;
    else
        Tmat = H_train' * UL;  % (n x m)*(m x L_eff) = n x L_eff

        cmul_C1 = n * m * L_eff;
        cadd_C1 = n * (m - 1) * L_eff;
        flop_C1 = 6 * cmul_C1 + 2 * cadd_C1;    % = 8nmL_eff - 2nL_eff

        invs = 1 ./ sL(:).';                    % 1 x L_eff (real)
        V_dense = Tmat .* invs;                 % column scaling

        flop_C2 = 2 * n * L_eff;                % real scalar * complex (per entry)
        flop_C  = flop_C1 + flop_C2;            % = 8nmL_eff
    end

    %% ---------------------------------------------------------------
    % Step D: Column-wise hard thresholding (zeroing) to build V_s
    %% ---------------------------------------------------------------
    Vs = zeros(n, L_eff, class(H_train));

    % Arithmetic FLOPs for magnitude^2: 3 FLOPs per entry
    flop_D_mag2 = 3 * n * L_eff;

    % Comparison model for top-M selection per column
    if strcmpi(opts.cmp_model, 'sort')
        % rough upper model: n*ceil(log2 n) comparisons per column
        cmp_D = L_eff * n * ceil(log2(max(n,2)));
    else
        % selection model: c_sel*n comparisons per column (expected)
        cmp_D = opts.c_sel * n * L_eff;
    end

    if L_eff > 0
        if M == 0
            % keep none (all zeros): already zero
        elseif M == n
            Vs = V_dense;
        else
            for i = 1:L_eff
                v = V_dense(:, i);
                mag2 = real(v).^2 + imag(v).^2;  % no sqrt

                % choose top-M indices (implementation convenience)
                [~, idx_keep] = maxk(mag2, M);

                vi = zeros(n, 1, class(H_train));
                vi(idx_keep) = v(idx_keep);
                Vs(:, i) = vi;
            end
        end
    end

    %% ---------------------------------------------------------------
    % Aggregate counts
    %% ---------------------------------------------------------------
    flop_B_total = flop_B_matmul_total + flop_B_qr_total + flop_B_final_GQ + flop_B_rayleigh + flop_B_ritz;
    flop_total_arith = flop_A + flop_B_total + flop_C + flop_D_mag2; % arithmetic only

    %% ---------------------------------------------------------------
    % Pack outputs
    %% ---------------------------------------------------------------
    out = struct();
    out.Vs      = Vs;
    out.sigma   = sL(:);
    out.V_dense = V_dense;
    out.UL      = UL;
    out.G       = G;

    out.flops = struct();
    out.flops.model = 'cmul=6, cadd=2, real*complex=2, sqrt=1, division=1';
    out.flops.stepA = struct('real_flops', flop_A, 'cmul', cmul_A, 'cadd', cadd_A, ...
                             'formula', '(4*n - 1)*m*(m+1)');
    out.flops.stepB = struct('real_flops_total', flop_B_total, ...
                             'matmul_each_iter', flop_B_matmul_each, ...
                             'matmul_total_loop', flop_B_matmul_total, ...
                             'qr_total_including_init', flop_B_qr_total, ...
                             'final_GQ', flop_B_final_GQ, ...
                             'rayleigh', flop_B_rayleigh, ...
                             'ritz_arith_only', flop_B_ritz, ...
                             'qr_detail', detail_QR_init);
    out.flops.stepC = struct('real_flops_total', flop_C, ...
                             'real_flops_matmul', flop_C1, ...
                             'real_flops_scaling', flop_C2, ...
                             'cmul_matmul', cmul_C1, 'cadd_matmul', cadd_C1, ...
                             'formula', '8*n*m*L_eff');
    out.flops.stepD = struct('real_flops_mag2', flop_D_mag2, ...
                             'note', 'Selection/ordering is comparison-dominated; comparisons are reported separately.');
    out.flops.total_arithmetic = flop_total_arith;

    out.comparisons = struct();
    out.comparisons.stepD = struct('model', opts.cmp_model, 'c_sel', opts.c_sel, 'count', cmp_D);
    out.comparisons.total = cmp_D;

    out.info = struct();
    out.info.N_train = m;
    out.info.N_feat  = n;
    out.info.gamma0  = kappa0;
    out.info.rho0    = rho0;
    out.info.L       = L;
    out.info.L_eff   = L_eff;
    out.info.M_keep  = M;
    out.info.T       = T;
    out.info.do_ritz = logical(opts.do_ritz);
end


% =====================================================================
function [Q, flop_qr, detail] = mgs_qr_count(X)
% Modified Gram-Schmidt QR with explicit FLOP counting (complex)
    [m, L] = size(X);
    V = X;
    Q = zeros(m, L, class(X));

    flop_qr = 0;
    flop_diag_total = 0;
    flop_pair_total = 0;

    for k = 1:L
        vk = V(:,k);

        % rkk_sq = real(vk'*vk)
        flop_qr = flop_qr + (8*m - 2);
        flop_diag_total = flop_diag_total + (8*m - 2);

        rkk = sqrt(real(vk' * vk));
        flop_qr = flop_qr + 1;
        flop_diag_total = flop_diag_total + 1;

        if rkk <= eps
            error('MGS breakdown: nearly linearly dependent columns.');
        end

        % invr = 1/rkk
        flop_qr = flop_qr + 1;
        flop_diag_total = flop_diag_total + 1;

        Q(:,k) = vk * (1/rkk);
        flop_qr = flop_qr + 2*m;
        flop_diag_total = flop_diag_total + 2*m;

        qk = Q(:,k);

        for j = k+1:L
            % rkj = qk' * V(:,j)
            flop_qr = flop_qr + (8*m - 2);
            flop_pair_total = flop_pair_total + (8*m - 2);

            % V(:,j) = V(:,j) - qk*rkj
            flop_qr = flop_qr + 8*m;
            flop_pair_total = flop_pair_total + 8*m;

            rkj = qk' * V(:,j); %#ok<NASGU>
            V(:,j) = V(:,j) - qk * (qk' * V(:,j));
        end
    end

    detail = struct();
    detail.diag_total = flop_diag_total;
    detail.pair_total = flop_pair_total;
    detail.closed_form = 'QR_MGS(m,L) = 10*m*L + (16*m-2)*L*(L-1)/2 (arithmetic only)';
end