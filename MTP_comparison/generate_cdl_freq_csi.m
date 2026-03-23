function Hf = generate_cdl_freq_csi(Ns, SCS, Nsub, TxArraySize, RxArraySize, delaySpread, cdlProfile, seed)
% Generate frequency-domain CSI tensor using 3GPP CDL channel
%
% Output:
%   Hf : [Ns, Nt, Nr, Nsub] complex
%
% Inputs:
%   Ns          : number of samples
%   SCS         : subcarrier spacing in kHz (e.g., 15, 30, 60, 120)
%   Nsub        : number of subcarriers in frequency axis
%                 (must be a multiple of 12 because MATLAB uses NSizeGrid in RBs)
%   TxArraySize : [M N P Mg Ng]
%   RxArraySize : [M N P Mg Ng]
%   delaySpread : RMS delay spread in seconds
%   cdlProfile  : 'CDL-A' ~ 'CDL-E'
%   seed        : single RNG seed for reproducibility
%
% TxArraySize / RxArraySize = [M N P Mg Ng]
%   M  : number of rows in each panel        (vertical direction)
%   N  : number of columns in each panel     (horizontal direction)
%   P  : number of polarizations (1 or 2)
%   Mg : number of panel rows                (vertical panel count)
%   Ng : number of panel columns             (horizontal panel count)
%
% Notes:
%   - First-arrival timing alignment is used.
%   - MaximumDopplerShift is fixed to 0 for quasi-static CSI.
%   - One slot is generated per sample, and the first OFDM symbol is used.

    arguments
        Ns (1,1) {mustBeInteger, mustBePositive}
        SCS (1,1) {mustBePositive}
        Nsub (1,1) {mustBeInteger, mustBePositive}
        TxArraySize (1,5) {mustBeInteger, mustBePositive}
        RxArraySize (1,5) {mustBeInteger, mustBePositive}
        delaySpread (1,1) {mustBePositive}
        cdlProfile (1,1) string
        seed (1,1) {mustBeInteger}
    end

    validProfiles = ["CDL-A","CDL-B","CDL-C","CDL-D","CDL-E"];
    assert(any(cdlProfile == validProfiles), 'cdlProfile must be one of CDL-A ~ CDL-E.');
    assert(mod(Nsub,12) == 0, 'Nsub must be a multiple of 12 (12 subcarriers per RB).');

    % Global reproducibility
    rng(seed, 'twister');

    % Carrier config
    carrier = nrCarrierConfig;
    carrier.SubcarrierSpacing = SCS;
    carrier.NSizeGrid = Nsub / 12;  % MATLAB expects RB count

    ofdmInfo = nrOFDMInfo(carrier);
    samplesPerSlot = sum(ofdmInfo.SymbolLengths(1:carrier.SymbolsPerSlot));

    % Infer Nt / Nr
    cdl0 = nrCDLChannel;
    cdl0.TransmitAntennaArray.Size = TxArraySize;
    cdl0.ReceiveAntennaArray.Size  = RxArraySize;
    info0 = info(cdl0);
    Nt = info0.NumInputSignals;
    Nr = info0.NumOutputSignals;
    release(cdl0);

    % Preallocate
    Hf = complex(zeros(Ns, Nt, Nr, Nsub));

    for i = 1:Ns
        % Configure channel
        cdl = nrCDLChannel;
        cdl.DelayProfile = cdlProfile;
        cdl.DelaySpread = delaySpread;
        cdl.MaximumDopplerShift = 0;
        cdl.ChannelFiltering = false;
        cdl.RandomStream = "Global stream";
        cdl.NumTimeSamples = samplesPerSlot;
        cdl.SampleRate = ofdmInfo.SampleRate;
        cdl.TransmitAntennaArray.Size = TxArraySize;
        cdl.ReceiveAntennaArray.Size  = RxArraySize;

        % Generate path gains
        [pathGains, sampleTimes] = cdl();
        pathFilters = getPathFilters(cdl);

        % First-arrival timing offset
        offset = localFirstArrivalOffset(pathGains, pathFilters, 1e-3);

        % Perfect channel estimate: [Nsub, Nsym, Nr, Nt]
        Hest = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);

        % Use first OFDM symbol only -> [Nsub, Nr, Nt]
        Hsym = squeeze(Hest(:,1,:,:));

        % Reorder to [Nt, Nr, Nsub]
        Hf(i,:,:,:) = permute(Hsym, [3 2 1]);

        release(cdl);
    end
end

function offset = localFirstArrivalOffset(pathGains, pathFilters, relThresh)
% First-arrival offset from earliest significant CIR sample

    if nargin < 3
        relThresh = 1e-3;
    end

    [~, mag] = nrPerfectTimingEstimate(pathGains, pathFilters);
    % mag: [Nh, Nr], after averaging over snapshots / summing Tx inside the function

    mag1 = max(mag, [], 2);   % keep earliest significant energy across Rx
    peakVal = max(mag1);

    if peakVal <= 0
        offset = 0;
        return;
    end

    th = relThresh * peakVal;
    idx = find(mag1 >= th, 1, 'first');

    if isempty(idx)
        offset = 0;
    else
        offset = idx - 1;     % sample offset
    end
end