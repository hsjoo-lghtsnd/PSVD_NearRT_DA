function Ht = CDL_convert_wrapper(H, Ntap)

SCS = 15;                      % kHz
Nsub = 624;                    % 52 RB * 12 subcarriers

% get Nfft
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = SCS;
carrier.NSizeGrid = Nsub / 12;
% ofdmInfo = nrOFDMInfo(carrier);
% Nfft = ofdmInfo.Nfft;

% Example CSI-RS
csirs = nrCSIRSConfig;
csirs.RowNumber = 1;
csirs.Density = 'three';
csirs.SymbolLocations = {6};
csirs.SubcarrierLocations = {0};

% Full-band channel tensor Hf: [Ns, Nt, Nr, 624]
[Hobs, ~, ~] = extract_csirs_observations(H, carrier, csirs);

Ht = freq_to_delay_csi(Hobs, Ntap);

end
