function [Hobs, scIdx, maskInfo] = extract_csirs_observations(Hf, carrier, csirs)
% Extract CSI-RS-observed subcarriers from full-band frequency-domain CSI.
%
% Input:
%   Hf      : [Ns, Nt, Nr, Nsub] complex
%             Full-band frequency-domain CSI over the active carrier grid.
%             Here, Nsub should match 12*carrier.NSizeGrid.
%
%   carrier : nrCarrierConfig object
%   csirs   : nrCSIRSConfig object
%
% Output:
%   Hobs    : [Ns, Nt, Nr, Nobs] complex
%             CSI observed only at subcarriers used by CSI-RS
%             (unique subcarrier locations across all CSI-RS REs in the slot)
%
%   scIdx   : [Nobs, 1] double
%             1-based subcarrier indices within the active carrier grid
%
%   maskInfo: struct
%             Additional information:
%               .maskRE        : [Nsub, Nsym, Nport] logical CSI-RS RE mask
%               .usedSymbols   : OFDM symbol indices touched by CSI-RS
%               .usedPorts     : port indices touched by CSI-RS
%               .linearIndices : original linear indices from nrCSIRSIndices
%
% Notes:
%   - This function extracts UNIQUE subcarrier positions only.
%   - Because Hf has no OFDM symbol axis, symbol-specific RE duplication is collapsed.
%   - If you want per-RE observations (subcarrier, symbol, port), you need a tensor
%     that also retains the OFDM symbol dimension.

    arguments
        Hf {mustBeNumeric}
        carrier
        csirs
    end

    assert(ndims(Hf) == 4, 'Hf must be [Ns, Nt, Nr, Nsub].');

    [Ns, Nt, Nr, Nsub] = size(Hf);
    expectedNsub = 12 * carrier.NSizeGrid;
    assert(Nsub == expectedNsub, ...
        'size(Hf,4) must equal 12*carrier.NSizeGrid.');

    % Build slot-grid size consistent with nrCSIRSIndices examples
    Nsym = carrier.SymbolsPerSlot;
    Nport = max(csirs.NumCSIRSPorts);

    % Generate linear CSI-RS RE indices on the slot grid
    ind = nrCSIRSIndices(carrier, csirs);   % linear indices into [Nsub, Nsym, Nport]

    % Create logical RE mask
    maskRE = false(Nsub, Nsym, Nport);
    maskRE(ind) = true;

    % Collapse over OFDM symbols and ports -> unique used subcarriers
    scMask = any(maskRE, [2 3]);            % [Nsub,1]
    scIdx = find(scMask);                   % 1-based active-grid subcarrier indices

    % Extract observations from Hf
    Hobs = Hf(:,:,:,scIdx);                 % [Ns, Nt, Nr, Nobs]

    % Extra info
    usedSymMask = squeeze(any(maskRE, [1 3]));   % [Nsym,1]
    usedPortMask = squeeze(any(maskRE, [1 2]));  % [Nport,1]

    maskInfo = struct();
    maskInfo.maskRE = maskRE;
    maskInfo.usedSymbols = find(usedSymMask);
    maskInfo.usedPorts = find(usedPortMask);
    maskInfo.linearIndices = ind;
end