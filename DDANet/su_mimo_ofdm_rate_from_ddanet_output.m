function [R, gamma, Psubband, Pcell, Wcell] = ...
    su_mimo_ofdm_rate_from_ddanet_output(Horg, H, Yhat, Nt, Nsband, Ptot, N0, opts)
%SU_MIMO_OFDM_RATE_FROM_DDANET_OUTPUT
% Rate evaluation using DDA-Net decoder output Yhat.
%
% Inputs
%   Horg    : [Nt, Nr, Nsub] true channel
%   H       : [Nt, Nr, Nsub] noisy CSIR
%   Yhat    : [2*Nt*Nsband, 1] real decoder output for ONE sample
%   Nt      : number of Tx antennas
%   Nsband  : number of subbands (e.g. 13)
%   Ptot    : total transmit power
%   N0      : noise variance
%
% opts
%   opts.subband_map     : [1, Nsub] mapping from subcarrier to subband
%   opts.use_pinv        : true/false
%   opts.reg_epsilon     : >=0
%   opts.return_cells    : true/false
%
% Outputs
%   R         : scalar spectral efficiency
%   gamma     : [1, Nsub] SINR
%   Psubband  : [Nt, 1, Nsband] reconstructed subband precoders
%   Pcell     : optional per-subcarrier precoders
%   Wcell     : optional per-subcarrier combiners

    arguments
        Horg {mustBeNumeric}
        H {mustBeNumeric}
        Yhat {mustBeNumeric}
        Nt (1,1) {mustBeInteger, mustBePositive}
        Nsband (1,1) {mustBeInteger, mustBePositive}
        Ptot (1,1) double {mustBePositive}
        N0 (1,1) double {mustBePositive}
        opts struct
    end

    if (~isfield(opts, 'use_pinv')) 
        opts.use_pinv=true;
    end
    if (~isfield(opts, 'reg_epsilon')) 
        opts.reg_epsilon = 0;
    end
    if (~isfield(opts, 'return_cells'))
        opts.return_cells = false; 
    end

    % Recover Vhat_stack: [Nt, Nsband, 1]
    Vhat = unpack_implicit_features(Yhat, Nt, Nsband, true);

    % Convert to precoder tensor [Nt, 1, Nsband]
    Psubband = complex(zeros(Nt, 1, Nsband));
    for s = 1:Nsband
        ps = Vhat(:,s,1);
        nrm = norm(ps);
        if nrm > 0
            ps = ps / nrm;
        end
        Psubband(:,1,s) = ps;
    end

    [R, gamma, Pcell, Wcell] = su_mimo_ofdm_rate_given_precoder( ...
        Horg, H, Psubband, 1, Ptot, N0, ...
        struct( ...
            'use_pinv', opts.use_pinv, ...
            'reg_epsilon', opts.reg_epsilon, ...
            'return_cells', opts.return_cells, ...
            'subband_map', opts.subband_map));
end