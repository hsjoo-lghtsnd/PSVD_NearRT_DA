Ns = 2000;
SCS = 15;                      % kHz
Nsub = 624;                    % 52 RB * 12 subcarriers
TxArraySize = [4 4 2 1 1];     % [rows, cols, pol, panelRows, panelCols]
RxArraySize = [2 2 1 1 1];
delaySpread = 300e-9;          % seconds
cdlProfile = "CDL-E";
seed = 20260226;

Ntap = 32;

%%%%

% Original frequency CSI
Hf = generate_cdl_freq_csi(Ns, SCS, Nsub, TxArraySize, RxArraySize, delaySpread, cdlProfile, seed);

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
[Hobs, scIdx, maskInfo] = extract_csirs_observations(Hf, carrier, csirs);

size(Hobs)      % [Ns, Nt, Nr, Nobs]
scIdx(1:10)     % first few CSI-RS-observed subcarrier indices
maskInfo.usedSymbols
maskInfo.usedPorts


Ht = freq_to_delay_csi(Hobs, Ntap);


Hobs_recon = delay_to_freq_csi(Ht, length(scIdx));

% Compare in frequency domain
plot_Hf_sample({Hobs, Hobs_recon}, 1, 1, 1, ...
    'Labels', {'Original H_f', 'Reconstructed from truncated H_t'}, ...
    'Scale', 'db', ...
    'TitleText', 'Frequency response comparison');

%%
Ntrain = 1000;
gamma0 = 0.02;
rho0 = 0.5;

% Rx = 1;
Rx = 1:4;

Htrain = reshape(Ht(1:Ntrain,:,Rx,:), Ntrain, []);
disp(size(Htrain));
Nfeat = size(Htrain,2);

L = min(floor(Nfeat*gamma0), Ntrain);

% SVD
[Uf, Sf, Vf] = svd(Htrain, 'econ');

VL = Vf(:,1:L);

Ntest = 1000;

Htest = reshape(Ht(Ns-Ntest+1:Ns, :, Rx, :), Ntest, []);
disp(size(Htest));

Z = Htest*VL;

rZ = zeros(size(Z,1), size(Z,2));
sigma = zeros(1,L);

% for i=1:size(Z,1)
%     rZ(i,:) = Z(i,:)/norm(Z(i,:), 'fro');
% end

for i=1:L
    sigma(i) = Sf(i,i);
end

% Zstat_mean = mean(abs(rZ),1);
% Zstat_std = std(abs(rZ),1);

Zstat_mean = mean(abs(Z),1);
Zstat_std = std(abs(Z),1);
sigma = sigma / sqrt(Ntrain);

%%

index = 1:length(Zstat_mean);


upper = Zstat_mean+Zstat_std;
lower = Zstat_mean-Zstat_std;

% upper = upper / sum(Zstat_mean);
% lower = lower / sum(Zstat_mean);
% Zstat_mean = Zstat_mean / sum(Zstat_mean);

figure;
hold on;
grid minor;
box on;

% shaded region
fill([index, fliplr(index)], [upper, fliplr(lower)], ...
     [0.7 0.85 1.0], ...      
     'EdgeColor', 'none', ...
     'FaceAlpha', 0.5);      

% mean line
plot(index, Zstat_mean, 'b-', 'LineWidth', 2);
plot(index, sigma, 'r--', 'LineWidth', 1);

set(gca, 'FontName', 'Times New Roman')
xlabel('Codeword Index', 'FontName', 'Times New Roman');
ylabel('Codeword Magnitude', 'FontName', 'Times New Roman');
ylim([0,2]);
xlim([1,L]);
legend('mean \pm stdv', 'mean', '\sigma/N^{1/2}_{train}');

set(gca,'FontName','Times New Roman',...
        'FontSize',16,...
        'LineWidth',1.2)

set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',16)