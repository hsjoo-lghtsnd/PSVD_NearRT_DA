% function sweepResult = run_ddanet_psvd_implicit_sweep()
%RUN_DDANET_PSVD_IMPLICIT_SWEEP
% Sweep field sample count Nfield in {20,50,100} and compare:
%   - DDA mismatch / adapted
%   - PSVD mismatch / adapted
%   - Oracle
%
% Uses the same 832-d implicit representation for both DDA and PSVD.
% Oracle shift timing is assumed.
%
% Checkpoints are saved under ./data as:
%   variableName-timestamp.mat
% or
%   bundleName-timestamp.mat
%
% Required helper functions:
%   - generate_cdl_freq_csi
%   - makeImplicit13SubbandInput
%   - createImCsiNetM832
%   - pretrainExistingImCsiNetForDDA
%   - buildPrestoredCodewordBank
%   - predictImCsiNetAutoencoder
%   - finetuneDDAImCsiNetS
%   - batch_eval_rate_from_ddanet_outputs
%   - psvd_codebook
%   - psvd_reconstruct_features
%   - avg_cosine_similarity_matrix
%   - make_contiguous_subband_map
%   - add_freq_csi_noise (optional)
%   - save_checkpoint_var
%   - save_checkpoint_bundle

%% ---------------------------------------------------------
% 0) Global config
% ---------------------------------------------------------
rng(20260318, 'twister');

dataDir = 'data';
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

SCS = 15;
Nsub = 624;
TxArraySize = [4 4 2 1 1];   % Nt = 32
RxArraySize = [1 1 2 1 1];   % Nr = 2
cdlProfile = "CDL-C";

Ns_source = 40000;
Ns_target = 10000;

delay_source = 300e-9;
delay_target = 30e-9;

Nt = 32;
Nsband = 13;
inputDim = 2 * Nt * Nsband;   % 832
compressRate = 1/64;          % encodedDim ~= 13

rho0 = 0.8;
Tpsvd = 30;

fieldList = [20 50 100];

usePerfectCSIR = true;
CSI_SNR_dB = 20;

% If canUseGPU is unavailable in your environment, replace with false
useGPU = canUseGPU;

% DDA pretraining options
preOpts = struct();
preOpts.MaxEpochs = 200;
preOpts.MiniBatchSize = 1024;
preOpts.InitialLearnRate = 1e-3;
preOpts.UseGPU = useGPU;
preOpts.Verbose = true;
preOpts.DoValidation = true;
preOpts.ValFrequency = 20;
preOpts.UseBinarization = false;   % unquantized comparison
preOpts.GradDecay = 0.9;
preOpts.SqGradDecay = 0.999;

% DDA finetuning base options
ddaBaseOpts = struct();
ddaBaseOpts.MaxEpochs = 100;
ddaBaseOpts.InitialLearnRate = 1e-2;
ddaBaseOpts.UseGPU = useGPU;
ddaBaseOpts.UseBinarization = false;  % unquantized comparison
ddaBaseOpts.AlphaDomainLoss = 50;
ddaBaseOpts.KernelSigma = 1.0;
ddaBaseOpts.GradDecay = 0.9;
ddaBaseOpts.SqGradDecay = 0.999;
ddaBaseOpts.Verbose = true;
ddaBaseOpts.ValFrequency = 5;

%% ---------------------------------------------------------
% 1) Generate full CSI source / target once
% ---------------------------------------------------------
fprintf('\n[1/7] Generating source/target CDL datasets...\n');

Hf_A = generate_cdl_freq_csi( ...
    Ns_source, SCS, Nsub, TxArraySize, RxArraySize, ...
    delay_source, cdlProfile, 20260318);
save_checkpoint_var('Hf_A', Hf_A, dataDir);

Hf_B = generate_cdl_freq_csi( ...
    Ns_target, SCS, Nsub, TxArraySize, RxArraySize, ...
    delay_target, cdlProfile, 20260319);
save_checkpoint_var('Hf_B', Hf_B, dataDir);

%% ---------------------------------------------------------
% 2) Build 832-d implicit datasets once
% ---------------------------------------------------------
fprintf('\n[2/7] Building implicit datasets...\n');

X_A = makeImplicit13SubbandInput(Hf_A);   % [832, Ns_source]
save_checkpoint_var('X_A', X_A, dataDir);

X_B = makeImplicit13SubbandInput(Hf_B);   % [832, Ns_target]
save_checkpoint_var('X_B', X_B, dataDir);

assert(size(X_A,1) == inputDim, 'X_A dimension mismatch.');
assert(size(X_B,1) == inputDim, 'X_B dimension mismatch.');

%% ---------------------------------------------------------
% 3) Split source / target once
% ---------------------------------------------------------
fprintf('\n[3/7] Splitting datasets...\n');

% Source split
N_A = size(X_A, 2);
idxA = randperm(N_A);
nA_tr = floor(0.8 * N_A);
nA_va = floor(0.1 * N_A);

idxA_train = idxA(1:nA_tr);
idxA_val   = idxA(nA_tr+1:nA_tr+nA_va);
idxA_test  = idxA(nA_tr+nA_va+1:end);

XA_train = X_A(:, idxA_train);
XA_val   = X_A(:, idxA_val);
XA_test  = X_A(:, idxA_test);
XSourceBank = XA_train;

% Target split
N_B = size(X_B, 2);
idxB = randperm(N_B);

% Keep a large pool so we can take 20/50/100 from the same prefix
idxB_fieldPool = idxB(1:100);
idxB_val       = idxB(101:1100);
idxB_test      = idxB(1101:end);

XB_fieldPool = X_B(:, idxB_fieldPool);
XB_val       = X_B(:, idxB_val);
XB_test      = X_B(:, idxB_test);

save_checkpoint_bundle('dataset_split', dataDir, ...
    'XA_train', XA_train, ...
    'XA_val', XA_val, ...
    'XA_test', XA_test, ...
    'XSourceBank', XSourceBank, ...
    'XB_fieldPool', XB_fieldPool, ...
    'XB_val', XB_val, ...
    'XB_test', XB_test, ...
    'idxA_train', idxA_train, ...
    'idxA_val', idxA_val, ...
    'idxA_test', idxA_test, ...
    'idxB_fieldPool', idxB_fieldPool, ...
    'idxB_val', idxB_val, ...
    'idxB_test', idxB_test);

%% ---------------------------------------------------------
% 4) Prepare target test Horg/H once
% ---------------------------------------------------------
fprintf('\n[4/7] Preparing target test channels for rate evaluation...\n');

HorgSet = permute(Hf_B(idxB_test,:,:,:), [2 3 4 1]);   % [Nt, Nr, Nsub, Ntest]

if usePerfectCSIR
    Hset = HorgSet;
else
    Hset = add_freq_csi_noise(HorgSet, CSI_SNR_dB);
end

save_checkpoint_bundle('target_rate_channels', dataDir, ...
    'HorgSet', HorgSet, ...
    'Hset', Hset);

%%

rateOpts = struct();
rateOpts.Nt = Nt;
rateOpts.Nsband = Nsband;
rateOpts.subband_map = make_contiguous_subband_map(Nsub, Nsband);
rateOpts.use_pinv = true;
rateOpts.reg_epsilon = 0;
rateOpts.return_cells = false;
rateOpts.verbose = true;
rateOpts.sample_indices = 1:size(HorgSet,4);

N0 = 1e-2;
SNR0_dB = 20;
Nlayer = 1;

Ptot = compute_ptot_for_target_snr0(HorgSet, Nlayer, N0, SNR0_dB);

fprintf('Computed Ptot for SNR0 = %.1f dB: %.6g\n', SNR0_dB, Ptot);

%% ---------------------------------------------------------
% 5) Pretrain DDA backbone once, and get DDA mismatch once
% ---------------------------------------------------------
fprintf('\n[5/7] Pretraining DDA backbone once...\n');

[encoderNet0, decoderNet0, config] = createImCsiNetM832( ...
    Nt, Nsband, compressRate, ...
    'HiddenDim1', 1664, ...
    'HiddenDim2', 832, ...
    'Leak', 0.3, ...
    'UseQuantLayer', false);

disp(config);

[encoderNet0, decoderNet0, preHist] = pretrainExistingImCsiNetForDDA( ...
    encoderNet0, decoderNet0, XA_train, XA_val, Nt, Nsband, preOpts);

save_checkpoint_bundle('pretrained_dda', dataDir, ...
    'encoderNet0', encoderNet0, ...
    'decoderNet0', decoderNet0, ...
    'preHist', preHist, ...
    'config', config);

ZSourceBank = buildPrestoredCodewordBank( ...
    encoderNet0, XSourceBank, preOpts.UseGPU, preOpts.UseBinarization);
save_checkpoint_var('ZSourceBank', ZSourceBank, dataDir);

% DDA mismatch output on target test
Yhat_dda_before = predictImCsiNetAutoencoder( ...
    encoderNet0, decoderNet0, XB_test, preOpts.UseGPU, preOpts.UseBinarization);
save_checkpoint_var('Yhat_dda_before', Yhat_dda_before, dataDir);

%%
res_dda_before = batch_eval_rate_from_ddanet_outputs( ...
    HorgSet, Hset, Yhat_dda_before, Ptot, N0, rateOpts);

cos_dda_before = avg_cosine_similarity_matrix(Yhat_dda_before, XB_test, Nt, Nsband);

save_checkpoint_bundle('dda_mismatch_result', dataDir, ...
    'res_dda_before', res_dda_before, ...
    'cos_dda_before', cos_dda_before);

%% ---------------------------------------------------------
% 6) PSVD mismatch once, Oracle once
% ---------------------------------------------------------
fprintf('\n[6/7] Computing PSVD mismatch and oracle once...\n');

[Vs_src, sigma_src, V_src, info_src] = psvd_codebook(XA_train.', compressRate, rho0, Tpsvd);

save_checkpoint_bundle('psvd_source', dataDir, ...
    'Vs_src', Vs_src, ...
    'sigma_src', sigma_src, ...
    'V_src', V_src, ...
    'info_src', info_src);

Yhat_psvd_before = psvd_reconstruct_features(XB_test, Vs_src, true);
save_checkpoint_var('Yhat_psvd_before', Yhat_psvd_before, dataDir);

res_psvd_before = batch_eval_rate_from_ddanet_outputs( ...
    HorgSet, Hset, Yhat_psvd_before, Ptot, N0, rateOpts);
cos_psvd_before = avg_cosine_similarity_matrix(Yhat_psvd_before, XB_test, Nt, Nsband);

save_checkpoint_bundle('psvd_mismatch_result', dataDir, ...
    'res_psvd_before', res_psvd_before, ...
    'cos_psvd_before', cos_psvd_before);

Yhat_oracle = XB_test;
save_checkpoint_var('Yhat_oracle', Yhat_oracle, dataDir);

res_oracle = batch_eval_rate_from_ddanet_outputs( ...
    HorgSet, Hset, Yhat_oracle, Ptot, N0, rateOpts);
cos_oracle = avg_cosine_similarity_matrix(Yhat_oracle, XB_test, Nt, Nsband);

save_checkpoint_bundle('oracle_result', dataDir, ...
    'res_oracle', res_oracle, ...
    'cos_oracle', cos_oracle);

%% ---------------------------------------------------------
% 7) Sweep over field sample counts
% ---------------------------------------------------------
fprintf('\n[7/7] Sweeping field sample counts...\n');

nCases = numel(fieldList);

R_dda_after = zeros(1, nCases);
R_psvd_after = zeros(1, nCases);
cos_dda_after = zeros(1, nCases);
cos_psvd_after = zeros(1, nCases);

ddaHistCell = cell(1, nCases);
YhatDDAafterCell = cell(1, nCases);
YhatPSVDafterCell = cell(1, nCases);
psvdInfoTgtCell = cell(1, nCases);

for i = 1:nCases
    Nfield = fieldList(i);
    fprintf('\n--- Field sample sweep: Nfield = %d ---\n', Nfield);

    XB_fieldN = XB_fieldPool(:, 1:Nfield);

    % ----- DDA adapted
    ddaOpts = ddaBaseOpts;
    ddaOpts.MiniBatchSize = Nfield;

    [encoderNetDDA, decoderNetDDA, ddaHist] = finetuneDDAImCsiNetS( ...
        encoderNet0, decoderNet0, ZSourceBank, XB_fieldN, XB_val, Nt, ddaOpts);

    save_checkpoint_bundle(sprintf('dda_adapted_Nfield_%d', Nfield), dataDir, ...
        'encoderNetDDA', encoderNetDDA, ...
        'decoderNetDDA', decoderNetDDA, ...
        'ddaHist', ddaHist, ...
        'Nfield', Nfield);

    Yhat_dda_after = predictImCsiNetAutoencoder( ...
        encoderNetDDA, decoderNetDDA, XB_test, ddaOpts.UseGPU, ddaOpts.UseBinarization);

    save_checkpoint_var(sprintf('Yhat_dda_after_Nfield_%d', Nfield), Yhat_dda_after, dataDir);

    res_dda_after = batch_eval_rate_from_ddanet_outputs( ...
        HorgSet, Hset, Yhat_dda_after, Ptot, N0, rateOpts);

    c_dda_after = avg_cosine_similarity_matrix(Yhat_dda_after, XB_test, Nt, Nsband);

    % ----- PSVD adapted
    [Vs_tgt, sigma_tgt, V_tgt, info_tgt] = psvd_codebook_real(XB_fieldN.', compressRate, rho0, Tpsvd);

    save_checkpoint_bundle(sprintf('psvd_target_Nfield_%d', Nfield), dataDir, ...
        'Vs_tgt', Vs_tgt, ...
        'sigma_tgt', sigma_tgt, ...
        'V_tgt', V_tgt, ...
        'info_tgt', info_tgt, ...
        'Nfield', Nfield);

    Yhat_psvd_after = psvd_reconstruct_features(XB_test, Vs_tgt, true);
    save_checkpoint_var(sprintf('Yhat_psvd_after_Nfield_%d', Nfield), Yhat_psvd_after, dataDir);

    res_psvd_after = batch_eval_rate_from_ddanet_outputs( ...
        HorgSet, Hset, Yhat_psvd_after, Ptot, N0, rateOpts);

    c_psvd_after = avg_cosine_similarity_matrix(Yhat_psvd_after, XB_test, Nt, Nsband);

    save_checkpoint_bundle(sprintf('rate_result_Nfield_%d', Nfield), dataDir, ...
        'res_dda_after', res_dda_after, ...
        'res_psvd_after', res_psvd_after, ...
        'c_dda_after', c_dda_after, ...
        'c_psvd_after', c_psvd_after, ...
        'Nfield', Nfield);

    % Store
    R_dda_after(i) = res_dda_after.R_mean;
    R_psvd_after(i) = res_psvd_after.R_mean;
    cos_dda_after(i) = c_dda_after;
    cos_psvd_after(i) = c_psvd_after;

    ddaHistCell{i} = ddaHist;
    YhatDDAafterCell{i} = Yhat_dda_after;
    YhatPSVDafterCell{i} = Yhat_psvd_after;
    psvdInfoTgtCell{i} = info_tgt;

    fprintf('Nfield=%d | DDA adapted R=%.6f, cos=%.6f | PSVD adapted R=%.6f, cos=%.6f\n', ...
        Nfield, R_dda_after(i), c_dda_after, R_psvd_after(i), c_psvd_after);
end

%% ---------------------------------------------------------
% Plot: average rate recovery curve
% ---------------------------------------------------------
figure;
hold on;
plot(fieldList, R_dda_after, '-o', 'LineWidth', 1.5, 'DisplayName','DDA adapted');
plot(fieldList, R_psvd_after, '-s', 'LineWidth', 1.5, 'DisplayName','PSVD adapted');
yline(res_dda_before.R_mean, '--', 'DisplayName','DDA mismatch');
yline(res_psvd_before.R_mean, '--', 'DisplayName','PSVD mismatch');
yline(res_oracle.R_mean, ':', 'DisplayName','Oracle');
xlabel('Number of fresh target samples');
ylabel('Average spectral efficiency (bps/Hz)');
title('Recovery curve versus field sample budget');
legend('Location','best');
grid on;
hold off;

%% ---------------------------------------------------------
% Plot: cosine similarity recovery curve
% ---------------------------------------------------------
figure;
hold on;
plot(fieldList, cos_dda_after, '-o', 'LineWidth', 1.5, 'DisplayName','DDA adapted');
plot(fieldList, cos_psvd_after, '-s', 'LineWidth', 1.5, 'DisplayName','PSVD adapted');
yline(cos_dda_before, '--', 'DisplayName','DDA mismatch');
yline(cos_psvd_before, '--', 'DisplayName','PSVD mismatch');
yline(cos_oracle, ':', 'DisplayName','Oracle');
xlabel('Number of fresh target samples');
ylabel('Average cosine similarity');
title('Implicit reconstruction recovery curve');
legend('Location','best');
grid on;
hold off;

%% ---------------------------------------------------------
% Pack outputs
% ---------------------------------------------------------
sweepResult = struct();

sweepResult.config = struct( ...
    'SCS', SCS, ...
    'Nsub', Nsub, ...
    'TxArraySize', TxArraySize, ...
    'RxArraySize', RxArraySize, ...
    'cdlProfile', cdlProfile, ...
    'delay_source', delay_source, ...
    'delay_target', delay_target, ...
    'compressRate', compressRate, ...
    'rho0', rho0, ...
    'Tpsvd', Tpsvd, ...
    'fieldList', fieldList);

sweepResult.pretraining = struct();
sweepResult.pretraining.config = config;
sweepResult.pretraining.preHist = preHist;
sweepResult.pretraining.ZSourceBank = ZSourceBank;

sweepResult.baselines = struct();
sweepResult.baselines.R_dda_before = res_dda_before.R_mean;
sweepResult.baselines.R_psvd_before = res_psvd_before.R_mean;
sweepResult.baselines.R_oracle = res_oracle.R_mean;
sweepResult.baselines.cos_dda_before = cos_dda_before;
sweepResult.baselines.cos_psvd_before = cos_psvd_before;
sweepResult.baselines.cos_oracle = cos_oracle;

sweepResult.curves = struct();
sweepResult.curves.fieldList = fieldList;
sweepResult.curves.R_dda_after = R_dda_after;
sweepResult.curves.R_psvd_after = R_psvd_after;
sweepResult.curves.cos_dda_after = cos_dda_after;
sweepResult.curves.cos_psvd_after = cos_psvd_after;

sweepResult.dda = struct();
sweepResult.dda.hist = ddaHistCell;
sweepResult.dda.Yhat_after = YhatDDAafterCell;

sweepResult.psvd = struct();
sweepResult.psvd.info_src = info_src;
sweepResult.psvd.info_tgt = psvdInfoTgtCell;
sweepResult.psvd.Yhat_after = YhatPSVDafterCell;

save_checkpoint_var('sweepResult', sweepResult, dataDir);

fprintf('\n=== Sweep Summary ===\n');
fprintf('Mismatch baselines: DDA R=%.6f, PSVD R=%.6f, Oracle R=%.6f\n', ...
    res_dda_before.R_mean, res_psvd_before.R_mean, res_oracle.R_mean);
for i = 1:nCases
    fprintf('Nfield=%3d | DDA R=%.6f, PSVD R=%.6f | DDA cos=%.6f, PSVD cos=%.6f\n', ...
        fieldList(i), R_dda_after(i), R_psvd_after(i), ...
        cos_dda_after(i), cos_psvd_after(i));
end


%% ---------------------------------------------------------
% 8) Additional native time-domain PSVD comparison
% ---------------------------------------------------------
fprintf('\n[8/8] Running native time-domain PSVD comparison...\n');

Ntap_td = 32;          % truncated taps in time domain
rho0_td = rho0;        % reuse sparsity
Tpsvd_td = Tpsvd;

% Two options:
%   (A) same compression ratio as implicit benchmark
compressRate_td = compressRate;
%   (B) or same latent dimension as implicit benchmark (L ~= 13)
% Nfeat_td = Nt * size(HorgSet,2) * Ntap_td;
% compressRate_td = 13 / Nfeat_td;

% ---------------------------------------------------------
% 8.1) Build source/target time-domain CSI
% ---------------------------------------------------------
Ht_A = freq_to_truncated_time_csi(Hf_A, Ntap_td);   % [Ns, Nt, Nr, Ntap]
Ht_B = freq_to_truncated_time_csi(Hf_B, Ntap_td);

save_checkpoint_var('Ht_A', Ht_A, dataDir);
save_checkpoint_var('Ht_B', Ht_B, dataDir);

XA_td = td_csi_to_feature_matrix(Ht_A);   % [Ns, Nfeat]
XB_td = td_csi_to_feature_matrix(Ht_B);   % [Ns, Nfeat]

save_checkpoint_var('XA_td', XA_td, dataDir);
save_checkpoint_var('XB_td', XB_td, dataDir);

% Use the SAME split indices as the implicit experiment
XA_td_train = XA_td(idxA_train, :);
XB_td_fieldPool = XB_td(idxB_fieldPool, :);
XB_td_test = XB_td(idxB_test, :);

save_checkpoint_bundle('td_dataset_split', dataDir, ...
    'XA_td_train', XA_td_train, ...
    'XB_td_fieldPool', XB_td_fieldPool, ...
    'XB_td_test', XB_td_test);

% ---------------------------------------------------------
% 8.2) Source PSVD codebook in time domain (mismatch baseline)
% ---------------------------------------------------------
[Vs_td_src, sigma_td_src, V_td_src, info_td_src] = ...
    psvd_codebook(XA_td_train, compressRate_td, rho0_td, Tpsvd_td);

save_checkpoint_bundle('psvd_td_source', dataDir, ...
    'Vs_td_src', Vs_td_src, ...
    'sigma_td_src', sigma_td_src, ...
    'V_td_src', V_td_src, ...
    'info_td_src', info_td_src);

% Reconstruct target test time-domain CSI using source codebook (mismatch)
Xhat_td_before = psvd_reconstruct_row_features(XB_td_test, Vs_td_src, true);   % [Ntest, Nfeat]
Ht_hat_before = feature_matrix_to_td_csi(Xhat_td_before, Nt, size(HorgSet,2), Ntap_td);
Hf_hat_before = truncated_time_to_freq_csi(Ht_hat_before, Nsub);

save_checkpoint_bundle('psvd_td_mismatch_recon', dataDir, ...
    'Xhat_td_before', Xhat_td_before, ...
    'Ht_hat_before', Ht_hat_before, ...
    'Hf_hat_before', Hf_hat_before);

% True target test channels
Horg_td_test = permute(Hf_B(idxB_test,:,:,:), [1 2 3 4]);  % [Ntest, Nt, Nr, Nsub]
Horg_td_test = permute(Horg_td_test, [2 3 4 1]);           % [Nt, Nr, Nsub, Ntest] for rate wrapper consistency if needed

% Imperfect CSIR for time-domain PSVD evaluation
if usePerfectCSIR
    H_td_test = Horg_td_test;
else
    H_td_test = add_freq_csi_noise(Horg_td_test, CSI_SNR_dB);
end
%%
% Evaluate mismatch rate + NMSE
[res_td_before, nmse_td_before] = batch_eval_rate_from_psvd_timedomain( ...
    permute(Hf_B(idxB_test,:,:,:), [2 3 4 1]), ...   % HorgSet [Nt, Nr, Nsub, Ntest]
    H_td_test, ...
    Hf_hat_before, ...
    Ptot, N0, Ntap_td, ...
    struct('use_pinv', true, 'reg_epsilon', 0, 'verbose', true));

save_checkpoint_bundle('psvd_td_mismatch_result', dataDir, ...
    'res_td_before', res_td_before, ...
    'nmse_td_before', nmse_td_before);

% ---------------------------------------------------------
% 8.3) Oracle matched reference in time domain
% ---------------------------------------------------------
% Oracle here means perfect reconstructed CSI = true truncated time-domain CSI
Ht_true_test = freq_to_truncated_time_csi(Hf_B(idxB_test,:,:,:), Ntap_td);
Hf_oracle_td = truncated_time_to_freq_csi(Ht_true_test, Nsub);

[res_td_oracle, nmse_td_oracle] = batch_eval_rate_from_psvd_timedomain( ...
    permute(Hf_B(idxB_test,:,:,:), [2 3 4 1]), ...
    H_td_test, ...
    Hf_oracle_td, ...
    Ptot, N0, Ntap_td, ...
    struct('use_pinv', true, 'reg_epsilon', 0, 'verbose', true));

save_checkpoint_bundle('psvd_td_oracle_result', dataDir, ...
    'res_td_oracle', res_td_oracle, ...
    'nmse_td_oracle', nmse_td_oracle);

% ---------------------------------------------------------
% 8.4) Sweep target-adapted time-domain PSVD
% ---------------------------------------------------------
R_td_after = zeros(1, nCases);
nmse_td_after = zeros(1, nCases);

VsTdTgtCell = cell(1, nCases);
infoTdTgtCell = cell(1, nCases);

for i = 1:nCases
    Nfield = fieldList(i);
    fprintf('\n--- Native time-domain PSVD: Nfield = %d ---\n', Nfield);

    XB_td_fieldN = XB_td_fieldPool(1:Nfield, :);

    [Vs_td_tgt, sigma_td_tgt, V_td_tgt, info_td_tgt] = ...
        psvd_codebook(XB_td_fieldN, compressRate_td, rho0_td, Tpsvd_td);

    save_checkpoint_bundle(sprintf('psvd_td_target_Nfield_%d', Nfield), dataDir, ...
        'Vs_td_tgt', Vs_td_tgt, ...
        'sigma_td_tgt', sigma_td_tgt, ...
        'V_td_tgt', V_td_tgt, ...
        'info_td_tgt', info_td_tgt, ...
        'Nfield', Nfield);

    Xhat_td_after = psvd_reconstruct_row_features(XB_td_test, Vs_td_tgt, true);
    Ht_hat_after = feature_matrix_to_td_csi(Xhat_td_after, Nt, size(HorgSet,2), Ntap_td);
    Hf_hat_after = truncated_time_to_freq_csi(Ht_hat_after, Nsub);

    save_checkpoint_bundle(sprintf('psvd_td_recon_Nfield_%d', Nfield), dataDir, ...
        'Xhat_td_after', Xhat_td_after, ...
        'Ht_hat_after', Ht_hat_after, ...
        'Hf_hat_after', Hf_hat_after, ...
        'Nfield', Nfield);

    [res_td_after, nmse_td_val] = batch_eval_rate_from_psvd_timedomain( ...
        permute(Hf_B(idxB_test,:,:,:), [2 3 4 1]), ...
        H_td_test, ...
        Hf_hat_after, ...
        Ptot, N0, Ntap_td, ...
        struct('use_pinv', true, 'reg_epsilon', 0, 'verbose', true));

    save_checkpoint_bundle(sprintf('psvd_td_result_Nfield_%d', Nfield), dataDir, ...
        'res_td_after', res_td_after, ...
        'nmse_td_val', nmse_td_val, ...
        'Nfield', Nfield);

    R_td_after(i) = res_td_after.R_mean;
    nmse_td_after(i) = nmse_td_val;

    VsTdTgtCell{i} = Vs_td_tgt;
    infoTdTgtCell{i} = info_td_tgt;

    fprintf('Nfield=%d | TD-PSVD adapted R=%.6f, NMSE=%.6f dB\n', ...
        Nfield, R_td_after(i), nmse_td_after(i));
end

% ---------------------------------------------------------
% 8.5) Plot native time-domain PSVD curves
% ---------------------------------------------------------
figure;
hold on;
plot(fieldList, R_td_after, '-d', 'LineWidth', 1.5, 'DisplayName','TD-PSVD adapted');
yline(res_td_before.R_mean, '--', 'DisplayName','TD-PSVD mismatch');
yline(res_td_oracle.R_mean, ':', 'DisplayName','TD-PSVD oracle');
xlabel('Number of fresh target samples');
ylabel('Average spectral efficiency (bps/Hz)');
title('Native time-domain PSVD recovery curve');
legend('Location','best');
grid on;
hold off;

figure;
hold on;
plot(fieldList, nmse_td_after, '-d', 'LineWidth', 1.5, 'DisplayName','TD-PSVD adapted');
yline(nmse_td_before, '--', 'DisplayName','TD-PSVD mismatch');
yline(nmse_td_oracle, ':', 'DisplayName','TD-PSVD oracle');
xlabel('Number of fresh target samples');
ylabel('Average NMSE (dB)');
title('Native time-domain PSVD NMSE curve');
legend('Location','best');
grid on;
hold off;

% Save into sweepResult if it already exists
if exist('sweepResult', 'var')
    sweepResult.timedomain_psvd = struct();
    sweepResult.timedomain_psvd.Ntap = Ntap_td;
    sweepResult.timedomain_psvd.compressRate = compressRate_td;
    sweepResult.timedomain_psvd.rho0 = rho0_td;
    sweepResult.timedomain_psvd.Tpsvd = Tpsvd_td;
    sweepResult.timedomain_psvd.R_before = res_td_before.R_mean;
    sweepResult.timedomain_psvd.R_oracle = res_td_oracle.R_mean;
    sweepResult.timedomain_psvd.R_after = R_td_after;
    sweepResult.timedomain_psvd.nmse_before = nmse_td_before;
    sweepResult.timedomain_psvd.nmse_oracle = nmse_td_oracle;
    sweepResult.timedomain_psvd.nmse_after = nmse_td_after;
    sweepResult.timedomain_psvd.Vs_tgt = VsTdTgtCell;
    sweepResult.timedomain_psvd.info_tgt = infoTdTgtCell;

    save_checkpoint_var('sweepResult', sweepResult, dataDir);
end


%%
R_perf_each = zeros(1, size(HorgSet,4));

for n = 1:size(HorgSet,4)
    Horg_n = HorgSet(:,:,:,n);
    R_perf_each(n) = su_mimo_ofdm_rate_imperfect( ...
        Horg_n, Horg_n, Horg_n, 1, Ptot, N0, ...
        struct('use_pinv', true, 'reg_epsilon', 0, 'return_cells', false));
end

R_perf_mean = mean(R_perf_each);

Yhat_oracle = XB_test;

res_oracle_perf = batch_eval_rate_from_ddanet_outputs( ...
    HorgSet, HorgSet, Yhat_oracle, Ptot, N0, rateOpts);



% end
