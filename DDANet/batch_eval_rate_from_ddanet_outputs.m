function results = batch_eval_rate_from_ddanet_outputs(HorgSet, Hset, YhatSet, Ptot, N0, opts)
%BATCH_EVAL_RATE_FROM_DDANET_OUTPUTS
% Evaluate average spectral efficiency over multiple DDA-Net decoder outputs.
%
% Inputs
%   HorgSet : [Nt, Nr, Nsub, Nsamples] complex
%   Hset    : [Nt, Nr, Nsub, Nsamples] complex
%   YhatSet : [2*Nt*Nsband, Nsamples] real
%   Ptot    : scalar, total transmit power
%   N0      : scalar, noise variance
%
% opts fields
%   .Nt               : number of Tx antennas
%   .Nsband           : number of subbands (e.g. 13)
%   .subband_map      : [1, Nsub] mapping from subcarrier to subband
%                       if empty, contiguous map is created automatically
%   .use_pinv         : true/false
%   .reg_epsilon      : >=0
%   .return_cells     : true/false
%   .verbose          : true/false
%   .sample_indices   : subset of samples to evaluate (default: all)
%
% Outputs
%   results is a struct with fields:
%       .R_each        : [1, Neval] rate per sample
%       .gamma_each    : [Nsub, Neval] if Nsym=1
%       .R_mean        : scalar average rate
%       .R_std         : scalar std of rate
%       .Psubband_cell : optional, 1xNeval cell
%       .Pcell         : optional, 1xNeval cell
%       .Wcell         : optional, 1xNeval cell
%
% Notes
%   - This wrapper assumes Nsym = 1, which is the natural DDA-Net setting
%     when the decoder output is a reconstructed dominant eigenvector per subband.
%   - It uses su_mimo_ofdm_rate_from_ddanet_output() internally.

    arguments
        HorgSet {mustBeNumeric}
        Hset {mustBeNumeric}
        YhatSet {mustBeNumeric}
        Ptot (1,1) double {mustBePositive}
        N0 (1,1) double {mustBePositive}
        opts struct = struct()
    end
    
    if ~isfield(opts, 'Nt') || isempty(opts.Nt)
        error('opts.Nt must be provided.');
    end
    if ~isfield(opts, 'Nsband') || isempty(opts.Nsband)
        error('opts.Nsband must be provided.');
    end
    
    if ~isfield(opts, 'subband_map') || isempty(opts.subband_map)
        opts.subband_map = [];
    end
    if ~isfield(opts, 'use_pinv') || isempty(opts.use_pinv)
        opts.use_pinv = true;
    end
    if ~isfield(opts, 'reg_epsilon') || isempty(opts.reg_epsilon)
        opts.reg_epsilon = 0;
    end
    if ~isfield(opts, 'return_cells') || isempty(opts.return_cells)
        opts.return_cells = false;
    end
    if ~isfield(opts, 'verbose') || isempty(opts.verbose)
        opts.verbose = true;
    end
    if ~isfield(opts, 'sample_indices') || isempty(opts.sample_indices)
        opts.sample_indices = [];
    end
    
    validateattributes(opts.Nt, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(opts.Nsband, {'numeric'}, {'scalar','integer','positive'});
    validateattributes(opts.reg_epsilon, {'double','single'}, {'scalar','nonnegative'});
    validateattributes(opts.use_pinv, {'logical','numeric'}, {'scalar'});
    validateattributes(opts.return_cells, {'logical','numeric'}, {'scalar'});
    validateattributes(opts.verbose, {'logical','numeric'}, {'scalar'});

    % Basic size checks
    [NtH, NrH, Nsub, Nsamples] = size(HorgSet);
    assert(all(size(Hset) == [NtH, NrH, Nsub, Nsamples]), ...
        'HorgSet and Hset must have the same size [Nt, Nr, Nsub, Nsamples].');
    assert(NtH == opts.Nt, 'opts.Nt does not match HorgSet size.');
    assert(size(YhatSet,1) == 2 * opts.Nt * opts.Nsband, ...
        'YhatSet first dimension must be 2*Nt*Nsband.');

    if isempty(opts.sample_indices)
        sampleIdx = 1:Nsamples;
    else
        sampleIdx = opts.sample_indices(:).';
        assert(all(sampleIdx >= 1 & sampleIdx <= Nsamples), ...
            'opts.sample_indices contains invalid sample indices.');
    end

    Neval = numel(sampleIdx);

    % Subband map
    if isempty(opts.subband_map)
        subband_map = make_contiguous_subband_map(Nsub, opts.Nsband);
    else
        subband_map = opts.subband_map;
        assert(numel(subband_map) == Nsub, ...
            'opts.subband_map must have length Nsub.');
    end

    % Preallocate
    R_each = zeros(1, Neval);
    gamma_each = zeros(Nsub, Neval);

    if opts.return_cells
        Psubband_cell = cell(1, Neval);
        Pcell_all = cell(1, Neval);
        Wcell_all = cell(1, Neval);
    else
        Psubband_cell = [];
        Pcell_all = [];
        Wcell_all = [];
    end

    % Per-sample evaluation
    for ii = 1:Neval
        n = sampleIdx(ii);

        Horg_sample = HorgSet(:,:,:,n);     % [Nt, Nr, Nsub]
        H_sample    = Hset(:,:,:,n);        % [Nt, Nr, Nsub]
        Yhat_sample = YhatSet(:,n);         % [2*Nt*Nsband, 1]

        localOpts = struct();
        localOpts.subband_map = subband_map;
        localOpts.use_pinv = opts.use_pinv;
        localOpts.reg_epsilon = opts.reg_epsilon;
        localOpts.return_cells = opts.return_cells;

        [Rtmp, gammatmp, Psubband, PcellTmp, WcellTmp] = ...
            su_mimo_ofdm_rate_from_ddanet_output( ...
                Horg_sample, H_sample, Yhat_sample, ...
                opts.Nt, opts.Nsband, Ptot, N0, localOpts);

        R_each(ii) = Rtmp;
        gamma_each(:,ii) = gammatmp(:);

        if opts.return_cells
            Psubband_cell{ii} = Psubband;
            Pcell_all{ii} = PcellTmp;
            Wcell_all{ii} = WcellTmp;
        end

        if opts.verbose && (mod(ii, max(1, floor(Neval/10))) == 0 || ii == Neval)
            fprintf('Evaluated %d / %d samples. Current mean rate = %.6f bps/Hz\n', ...
                ii, Neval, mean(R_each(1:ii)));
        end
    end

    % Summary
    results = struct();
    results.sample_indices = sampleIdx;
    results.R_each = R_each;
    results.gamma_each = gamma_each;
    results.R_mean = mean(R_each);
    results.R_std = std(R_each);

    if opts.return_cells
        results.Psubband_cell = Psubband_cell;
        results.Pcell = Pcell_all;
        results.Wcell = Wcell_all;
    end
end