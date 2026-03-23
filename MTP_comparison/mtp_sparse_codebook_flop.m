function [Vs, Ti, flop, info] = mtp_sparse_codebook_flop(Htrain, gamma0, rho0, tol)
%MTP_SPARSE_CODEBOOK_FLOP  MTP with ADD/MUL counting (model-based)
%
% Inputs:
%   Htrain : [Ntrain x Nfeat] complex (or real)
%   gamma0 : sets L = floor(Nfeat * gamma0)
%   rho0   : sets k = floor((1-rho0) * Nfeat)
%   tol    : break if NMSE(x_new, x_old) < tol
%
% Outputs:
%   Vs   : [Nfeat x L]
%   Ti   : [L x 1]
%   flop : struct with fields .ADD .MUL (counters)
%   info : diagnostics struct
%
% Counting rules:
%   real add/sub: ADD += 1
%   real mul:     MUL += 1
%   complex add/sub: ADD += 2
%   complex mul:     MUL += 4, ADD += 2  (as you specified)

    if nargin < 4
        error('Usage: [Vs, Ti, flop] = mtp_sparse_codebook_flop(Htrain, gamma0, rho0, tol)');
    end

    % ---- settings ----
    maxIter = 500;
    use_cov_scale = true;       % R = (H^H H)/Ntrain
    reorth_after_trunc = false; % optional
    % ------------------

    [Ntrain, d] = size(Htrain);

    L = floor(d * gamma0);
    L = max(1, min(d, L));

    k = floor((1 - rho0) * d);
    k = max(1, min(d, k));

    % FLOP counters
    ADD = 0;
    MUL = 0;

    % ---- Form R = H^H H ----
    % Each entry is dot-product length Ntrain:
    %   N complex mul + (N-1) complex add
    % complex mul -> MUL+4, ADD+2
    % complex add -> ADD+2
    % Per entry:
    %   MUL += 4N
    %   ADD += 2N (from mul) + 2(N-1) (from adds) = 4N-2
    % Total entries: d^2
    R = Htrain' * Htrain;
    MUL = MUL + 4 * Ntrain * d^2;
    ADD = ADD + (4 * Ntrain - 2) * d^2;

    if use_cov_scale && Ntrain > 0
        R = R / Ntrain;
        % Scaling by real scalar: each complex element -> 2 real mul
        MUL = MUL + 2 * d^2;
    end

    % Symmetrize: R = (R + R')/2
    R = (R + R') / 2;
    % R + R' : d^2 complex adds => ADD += 2 d^2
    ADD = ADD + 2 * d^2;
    % /2 scaling: 2 real mul per complex entry => MUL += 2 d^2
    MUL = MUL + 2 * d^2;

    Vs = zeros(d, L, class(Htrain));
    Ti = zeros(L, 1);
    nmse_last = zeros(L, 1);
    converged = false(L, 1);

    for i = 1:L
        % init x
        if isreal(Htrain)
            x = randn(d,1);
        else
            x = randn(d,1) + 1j*randn(d,1);
        end
        % normalize x (count norm + scaling)
        [ADD, MUL] = count_norm_and_scale(d, ADD, MUL);
        x = x / max(norm(x,2), eps);

        for t = 1:maxIter
            x_old = x;

            % ---- x = R * x ----
            x = R * x;
            % matvec: d dot-products length d
            % per output: MUL += 4d, ADD += 4d-2
            % total: MUL += 4d^2, ADD += d(4d-2)=4d^2-2d
            MUL = MUL + 4 * d^2;
            ADD = ADD + (4 * d^2 - 2 * d);

            % ---- orthogonalize against previous Vs ----
            if i > 1
                for j = 1:(i-1)
                    vj = Vs(:, j);

                    % alpha = vj' * x  (dot length d)
                    MUL = MUL + 4 * d;
                    ADD = ADD + (4 * d - 2);
                    alpha = vj' * x;

                    % x = x - vj * alpha
                    % vj*alpha: d complex mul => MUL += 4d, ADD += 2d
                    % subtraction: d complex sub => ADD += 2d
                    MUL = MUL + 4 * d;
                    ADD = ADD + 4 * d;
                    x = x - vj * alpha;
                end
            end

            % ---- normalize x ----
            [ADD, MUL] = count_norm_and_scale(d, ADD, MUL);
            nx = norm(x,2);
            if nx <= eps
                % restart
                if isreal(Htrain)
                    x = randn(d,1);
                else
                    x = randn(d,1) + 1j*randn(d,1);
                end
                [ADD, MUL] = count_norm_and_scale(d, ADD, MUL);
                x = x / max(norm(x,2), eps);
                continue;
            end
            x = x / nx;
            % scaling by real scalar: 2 real mul per complex entry
            MUL = MUL + 2 * d;

            % ---- truncation: keep k largest magnitude ----
            idx = topk_indices(abs(x), k);
            xt = zeros(d, 1, class(Htrain));
            xt(idx) = x(idx);
            % assignments not counted

            % optional re-orth after truncation
            if reorth_after_trunc && i > 1
                for j = 1:(i-1)
                    vj = Vs(:, j);
                    MUL = MUL + 4 * d;
                    ADD = ADD + (4 * d - 2);
                    alpha = vj' * xt;
                    MUL = MUL + 4 * d;
                    ADD = ADD + 4 * d;
                    xt = xt - vj * alpha;
                end
            end

            % normalize xt (k-sparse, but stored dense; count using k)
            [ADD, MUL] = count_norm_and_scale(k, ADD, MUL);
            nxt = norm(xt,2);
            if nxt <= eps
                [~, imax] = max(abs(x));
                xt = zeros(d,1, class(Htrain));
                xt(imax) = x(imax);
                % norm of 1-sparse:
                [ADD, MUL] = count_norm_and_scale(1, ADD, MUL);
                nxt = norm(xt,2);
            end
            x = xt / nxt;
            MUL = MUL + 2 * d; % scale dense vector by real scalar

            % ---- NMSE break: ||x - x_old||^2 / ||x_old||^2 ----
            % dx = x - x_old : d complex sub => ADD += 2d
            ADD = ADD + 2 * d;
            dx = x - x_old;

            % ||dx||^2 : treat as sum of |dx|^2; model each |.|^2 as 1 complex mul
            % MUL += 4d, ADD += 2d (from mul) + (d-1) real adds
            MUL = MUL + 4 * d;
            ADD = ADD + 2 * d + (d - 1);

            % ||x_old||^2 similarly
            MUL = MUL + 4 * d;
            ADD = ADD + 2 * d + (d - 1);

            denom = max(norm(x_old,2)^2, eps);
            nmse = norm(dx,2)^2 / denom;

            % division not counted (implementation-dependent)

            if nmse < tol
                Ti(i) = t;
                nmse_last(i) = nmse;
                converged(i) = true;
                break;
            end

            if t == maxIter
                Ti(i) = t;
                nmse_last(i) = nmse;
            end
        end

        Vs(:, i) = x;
    end

    flop = struct('ADD', ADD, 'MUL', MUL, 'TOTAL', ADD + MUL);
    info = struct();
    info.Ntrain = Ntrain;
    info.Nfeat  = d;
    info.L      = L;
    info.k      = k;
    info.tol    = tol;
    info.maxIter = maxIter;
    info.nmse_last = nmse_last;
    info.converged = converged;
    info.use_cov_scale = use_cov_scale;
    info.reorth_after_trunc = reorth_after_trunc;
end

function [ADD, MUL] = count_norm_and_scale(n, ADD, MUL)
% Count operations to compute norm(x,2) for n-length complex vector (model)
% We model |x_t|^2 as 1 complex multiply: conj(x_t)*x_t
% For each element:
%   complex mul: MUL += 4, ADD += 2
% Summation: (n-1) real adds: ADD += (n-1)
% (sqrt) ignored
%
% Note: This is a conservative/consistent model; true cost can be lower.

    if n <= 0
        return;
    end
    MUL = MUL + 4 * n;
    ADD = ADD + 2 * n + (n - 1);
end

function idx = topk_indices(v, k)
    n = numel(v);
    k = max(1, min(n, k));
    if exist('maxk', 'file') == 2
        [~, idx] = maxk(v, k);
    else
        [~, ord] = sort(v, 'descend');
        idx = ord(1:k);
    end
end