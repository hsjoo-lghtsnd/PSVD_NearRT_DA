function [results, meanNmseDb] = batch_eval_rate_from_psvd_timedomain(HorgSet, Hset, HtildeSet, Ptot, N0, Ntap, opts)
% batch_eval_rate_from_psvd_timedomain
%
% Input:
%   HorgSet   : [Nt, Nr, Nsub, Nsamples] true channel
%   Hset      : [Nt, Nr, Nsub, Nsamples] imperfect CSIR
%   HtildeSet : [Nsamples, Nt, Nr, Nsub] OR [Nt, Nr, Nsub, Nsamples]
%               reconstructed CSIT from time-domain PSVD
%   Ptot      : scalar
%   N0        : scalar
%   Ntap      : kept only for metadata / consistency
%
% opts:
%   .use_pinv     : logical
%   .reg_epsilon  : scalar
%   .verbose      : logical
%
% Output:
%   results:
%       .R_each
%       .R_mean
%       .R_std
%       .nmse_each_db
%   meanNmseDb:
%       scalar mean NMSE in dB

    arguments
        HorgSet {mustBeNumeric}
        Hset {mustBeNumeric}
        HtildeSet {mustBeNumeric}
        Ptot (1,1) double {mustBePositive}
        N0 (1,1) double {mustBePositive}
        Ntap (1,1) {mustBeInteger, mustBePositive}
        opts struct = struct()
    end
    
    if ~isfield(opts, 'use_pinv') || isempty(opts.use_pinv)
        opts.use_pinv = true;
    end
    if ~isfield(opts, 'reg_epsilon') || isempty(opts.reg_epsilon)
        opts.reg_epsilon = 0;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = true;
    end
    
    validateattributes(opts.use_pinv, {'logical','numeric'}, {'scalar'});
    validateattributes(opts.reg_epsilon, {'double','single'}, {'scalar','nonnegative'});
    validateattributes(opts.verbose, {'logical','numeric'}, {'scalar'});

    [Nt, Nr, Nsub, Nsamples] = size(HorgSet);
    assert(all(size(Hset) == [Nt, Nr, Nsub, Nsamples]), ...
        'HorgSet and Hset must have same size [Nt, Nr, Nsub, Nsamples].');

    % Accept either [Ns, Nt, Nr, Nsub] or [Nt, Nr, Nsub, Ns]
    szHtilde = size(HtildeSet);
    if isequal(szHtilde, [Nsamples, Nt, Nr, Nsub])
        HtildeEval = permute(HtildeSet, [2 3 4 1]);   % -> [Nt, Nr, Nsub, Ns]
    elseif isequal(szHtilde, [Nt, Nr, Nsub, Nsamples])
        HtildeEval = HtildeSet;
    else
        error('HtildeSet must be [Nsamples, Nt, Nr, Nsub] or [Nt, Nr, Nsub, Nsamples].');
    end

    R_each = zeros(1, Nsamples);
    nmse_each_db = zeros(1, Nsamples);

    for n = 1:Nsamples
        Horg = HorgSet(:,:,:,n);
        H    = Hset(:,:,:,n);
        Htilde = HtildeEval(:,:,:,n);

        [Rtmp, ~] = su_mimo_ofdm_rate_imperfect( ...
            Horg, H, Htilde, 1, Ptot, N0, ...
            struct('use_pinv', opts.use_pinv, ...
                   'reg_epsilon', opts.reg_epsilon, ...
                   'return_cells', false));

        R_each(n) = Rtmp;

        num = sum(abs(Horg(:) - Htilde(:)).^2);
        den = sum(abs(Horg(:)).^2) + 1e-12;
        nmse_each_db(n) = 10 * log10(num / den);

        if opts.verbose && (mod(n, max(1, floor(Nsamples/10))) == 0 || n == Nsamples)
            fprintf('TD-PSVD eval %d / %d, current mean R = %.6f, mean NMSE = %.6f dB\n', ...
                n, Nsamples, mean(R_each(1:n)), mean(nmse_each_db(1:n)));
        end
    end

    results = struct();
    results.Ntap = Ntap;
    results.R_each = R_each;
    results.R_mean = mean(R_each);
    results.R_std = std(R_each);
    results.nmse_each_db = nmse_each_db;
    results.nmse_mean_db = mean(nmse_each_db);

    meanNmseDb = results.nmse_mean_db;
end