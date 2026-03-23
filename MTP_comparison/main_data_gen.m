Ns = 2000;
SCS = 15;                      % kHz
Nsub = 624;                    % 52 RB * 12 subcarriers
TxArraySize = [4 4 2 1 1];     % [rows, cols, pol, panelRows, panelCols]
RxArraySize = [2 2 1 1 1];
delaySpread = 300e-9;          % seconds
cdlProfile = "CDL-B";
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
Ntrain = 100;
kappa0 = 0.02;
rho0 = 0.9;

% Rx = 1;
Rx = 1:4;

Htrain = reshape(Ht(1:Ntrain,:,Rx,:), Ntrain, []);
disp(size(Htrain));
Nfeat = size(Htrain,2);

L = min(floor(Nfeat*kappa0), Ntrain);
T = 30;
opts = struct('cmp_model','select','c_sel',3,'do_ritz',true);

out = psvd_codebook_flop(Htrain, kappa0, rho0, T);

fprintf('L=%d, L_eff=%d, M_keep=%d\n', out.info.L, out.info.L_eff, out.info.M_keep);
fprintf('Arithmetic FLOPs total = %.0f\n', out.flops.total_arithmetic);
fprintf('Step D comparisons (%s) = %.0f\n', out.comparisons.stepD.model, out.comparisons.stepD.count);

disp(size(out.Vs));
disp(out.sigma.');

VL = out.Vs;
sigma = out.sigma;


[Ue, Se, Ve] = svd(Htrain, 'econ');

Lcmp = min(numel(sigma), L);
s_ref = diag(Se);
s_ref = s_ref(1:Lcmp);

rel_err_s = norm(sigma(1:Lcmp) - s_ref) / max(norm(s_ref), eps);

Vref = Ve(:,1:Lcmp);
proj_err = norm(Vref*Vref' - VL(:,1:Lcmp)*VL(:,1:Lcmp)', 'fro') ...
           / max(norm(Vref*Vref','fro'), eps);

fprintf('Relative singular-value error = %.3e\n', rel_err_s);
fprintf('Right-subspace projector error = %.3e\n', proj_err);


[Vs2, Ti2, flop, info2] = mtp_sparse_codebook_flop(Htrain, kappa0, rho0, 1e-6);
disp(flop)
fprintf('Mean Ti: %.2f, Max Ti: %d\n', mean(Ti2), max(Ti2));


Ntest = 1000;

Htest = reshape(Ht(Ns-Ntest+1:Ns, :, Rx, :), Ntest, []);
disp(size(Htest));

% MTP Vs^H
[mres1, mres2, mres3] = my_print_error(Htest, Htest*Vs2*Vs2');
% MTP pinv(Vs)
[mres4, mres5, mres6] = my_print_error(Htest, Htest*Vs2*pinv(Vs2));

% PSVD Vs^H
[sres1, sres2, sres3] = my_print_error(Htest, Htest*VL*VL');
% PSVD pinv(Vs)
[sres4, sres5, sres6] = my_print_error(Htest, Htest*VL*pinv(VL));
