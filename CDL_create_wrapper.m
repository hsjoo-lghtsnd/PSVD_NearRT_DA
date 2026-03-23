function Horg = CDL_create_wrapper(Ntot, cdlProfile, delaySpread, seed)
% cdlProfile = 'CDL-A' to 'CDL-E'
% delaySpread in seconds
SCS = 15;                      % kHz
Nsub = 624;                    % 52 RB * 12 subcarriers
TxArraySize = [4 4 2 1 1];     % [rows, cols, pol, panelRows, panelCols]
RxArraySize = [2 2 1 1 1];

% Original frequency CSI
Horg = generate_cdl_freq_csi(Ntot, SCS, Nsub, TxArraySize, RxArraySize, delaySpread, cdlProfile, seed);

end