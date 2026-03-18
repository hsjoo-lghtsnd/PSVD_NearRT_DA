function [R, gamma, Pcell, Wcell] = su_mimo_ofdm_rate_given_precoder(Horg, H, Pgiven, Nsym, Ptot, N0, opts)
%SU_MIMO_OFDM_RATE_GIVEN_PRECODER
% Compute achievable spectral efficiency R for SU-MIMO-OFDM when the
% precoder is given directly (e.g., by DDA-Net / ImCsiNet implicit feedback).
%
% Inputs
%   Horg    : [Nt, Nr, Nsub] complex, true channel per subcarrier
%   H       : [Nt, Nr, Nsub] complex, noisy CSIR per subcarrier
%   Pgiven  : either
%               [Nt, Nsym, Nsub]   precoder per subcarrier
%             or
%               [Nt, Nsym, Nsband] precoder per subband
%
%   Nsym    : number of spatial layers (for DDA-Net usually Nsym = 1)
%   Ptot    : total transmit power across all Tx antennas
%   N0      : noise variance per receive antenna
%
% opts
%   opts.use_pinv        : true/false (default: true)
%   opts.reg_epsilon     : >=0 (default: 0)
%   opts.return_cells    : true/false (default: false)
%   opts.subband_map     : [1, Nsub] integer map from subcarrier to subband index
%                          required if size(Pgiven,3) ~= Nsub
%
% Outputs
%   R       : scalar spectral efficiency averaged over subcarriers
%   gamma   : [Nsym, Nsub] SINR per layer/subcarrier
%   Pcell   : 1xNsub cell of precoders (optional)
%   Wcell   : 1xNsub cell of combiners (optional)

    arguments
        Horg {mustBeNumeric}
        H {mustBeNumeric}
        Pgiven {mustBeNumeric}
        Nsym (1,1) {mustBeInteger, mustBePositive}
        Ptot (1,1) double {mustBePositive}
        N0 (1,1) double {mustBePositive}
        opts.use_pinv (1,1) logical = true
        opts.reg_epsilon (1,1) double {mustBeNonnegative} = 0
        opts.return_cells (1,1) logical = false
        opts.subband_map = []
    end

    % Dimensions
    [Nt, Nr, Nsub] = size(Horg);
    assert(all(size(H) == [Nt, Nr, Nsub]), ...
        'Horg and H must have the same size [Nt, Nr, Nsub].');
    assert(Nsym <= min(Nt, Nr), 'Nsym must be <= min(Nt, Nr).');

    % Internally use [Nr, Nt, Nsub]
    Horg_rtn = permute(Horg, [2 1 3]);   % [Nr, Nt, Nsub]
    H_rtn    = permute(H,    [2 1 3]);   % [Nr, Nt, Nsub]

    % Precoder storage mode
    [NtP, NsymP, Np] = size(Pgiven);
    assert(NtP == Nt, 'First dimension of Pgiven must equal Nt.');
    assert(NsymP == Nsym, 'Second dimension of Pgiven must equal Nsym.');

    useSubbandMap = (Np ~= Nsub);
    if useSubbandMap
        assert(~isempty(opts.subband_map), ...
            'opts.subband_map is required when Pgiven is not given per subcarrier.');
        assert(numel(opts.subband_map) == Nsub, ...
            'opts.subband_map must have length Nsub.');
        assert(all(opts.subband_map >= 1) && all(opts.subband_map <= Np), ...
            'opts.subband_map contains invalid subband indices.');
    end

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
        Horg_k = Horg_rtn(:,:,k);   % Nr x Nt
        H_k    = H_rtn(:,:,k);      % Nr x Nt

        % ---- Given precoder
        if useSubbandMap
            sb = opts.subband_map(k);
            Pk = Pgiven(:,:,sb);    % Nt x Nsym
        else
            Pk = Pgiven(:,:,k);     % Nt x Nsym
        end

        % Normalize ||Pk||_F^2 = Nsym
        fro2 = sum(abs(Pk(:)).^2);
        if fro2 > 0
            Pk = Pk * sqrt(Nsym / fro2);
        end

        % ---- ZF combiner from noisy CSIR
        Heff = H_k * Pk;            % Nr x Nsym
        G = Heff' * Heff;           % Nsym x Nsym

        if opts.reg_epsilon > 0
            G = G + opts.reg_epsilon * eye(Nsym);
        end

        if opts.use_pinv
            Wk = Heff * pinv(G);
        else
            Wk = Heff / G;
        end

        % ---- Effective true channel after precoding/combining
        Gtrue = Wk' * Horg_k * Pk;  % Nsym x Nsym

        diagPow = abs(diag(Gtrue)).^2;
        totalRowPow = sum(abs(Gtrue).^2, 2);
        interfPow = totalRowPow - diagPow;
        wNorm2 = sum(abs(Wk).^2, 1).';

        gamma(:,k) = (eta * diagPow) ./ (eta * interfPow + N0 * wNorm2);

        if opts.return_cells
            Pcell{k} = Pk;
            Wcell{k} = Wk;
        end
    end

    R = mean(sum(log2(1 + gamma), 1));
end