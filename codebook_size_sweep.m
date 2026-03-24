clc; clear;

%% ====== USER SETTINGS ======
snrCsi_dB = 20;               % CSI-RS SNR definition used for Horg -> H
csir_mode = 'global';         % 'global' | 'perSample' | 'perSampleSub'

SNR0_dB = 20;

% Sweep settings
kappa0_list = [0.001 0.003 0.005 0.007 0.01];
rho0_list   = [0.80 0.85 0.90 0.93 0.95 0.97 0.98];

Ntrain = 100;
Ntest = 1000;

E=2; Escenario=5;
Eprime=2; EprimeScenario=1;

T = 30; % PSVD iteration

%% ====== (1) Horg -> H (imperfect CSIR), Transform to time-domain ======

Ntot = Ntrain + Ntest;

E_opts = struct;
E_opts.seed = 12345;
[Ht, Horg] = load_environment(E, Ntot, Escenario, E_opts);

[H, sigma2_csi] = simulate_imperfect_csir_batch(Horg, snrCsi_dB, csir_mode);
if (E == 2)
    Ntap = size(Ht,4);
    Ht = CDL_convert_wrapper(H, Ntap);
else
    Ntap = size(Ht,4);
    Ht = freq_to_delay_csi(H, Ntap);
end
fprintf('Train environment load completed: E%d, scenario %d, SNRcsi-dB = %f\n', E, Escenario, snrCsi_dB);

Htrain = reshape(Ht(1:Ntrain,:,:,:), Ntrain, []);

if ~((E==Eprime) && (Escenario==EprimeScenario))
    E_opts.seed = 56789;
    [Htprime, Horgprime] = load_environment(Eprime, Ntot, EprimeScenario, E_opts);

    [Hprime, sigma2_csi_prime] = simulate_imperfect_csir_batch(Horgprime, snrCsi_dB, csir_mode);
    if (Eprime == 2)
        Ntap = size(Htprime,4);
        Htprime = CDL_convert_wrapper(Hprime, Ntap);
    else
        Ntap = size(Htprime,4);
        Htprime = freq_to_delay_csi(Hprime, Ntap);
    end
    fprintf('Test environment load completed: E%d, scenario %d, SNRcsi-dB = %f\n', Eprime, EprimeScenario, snrCsi_dB);
else
    fprintf('No domain shift specified: using training environment as test, but separately allocating.\n');
    Htprime = Ht;
    Hprime = H;
    Horgprime = Horg;
end

Htrain2 = reshape(Htprime(1:Ntrain,:,:,:), Ntrain, []);
Htest   = reshape(Htprime(Ntrain+1:Ntrain+Ntest,:,:,:), Ntest, []);

%% ====== Common parameters for rate evaluation ======

SNR0 = 10^(SNR0_dB/10);
Nt = size(Horg, 2);
Nr = size(Horg, 3);
Nlayer_max = max(min(Nt, Nr), 1);
N0   = 1e-2;
Nsub = size(Horg, 4);

PL = mean(abs(Horg(:)).^2);

Horg_test = Horgprime(Ntrain+1:Ntrain+Ntest,:,:,:);
H_test    = Hprime(Ntrain+1:Ntrain+Ntest,:,:,:);

Nfeat = Nt * Nr * Ntap;

opts = struct('use_pinv', true, 'reg_epsilon', 1e-6, 'return_cells', false);

%% ====== Sweep result containers ======

Nk = numel(kappa0_list);
Nrho = numel(rho0_list);

nnz_mat       = nan(Nk, Nrho);                  % x-axis = L*M
L_mat         = nan(Nk, Nrho);
M_mat         = nan(Nk, Nrho);

RavgC_mat     = nan(Nk, Nrho, Nlayer_max);     % R(E', C)
RavgCp_mat    = nan(Nk, Nrho, Nlayer_max);     % R(E', C')
RperfC_mat    = nan(Nk, Nrho, Nlayer_max);     % perfect/reference
RperfCp_mat   = nan(Nk, Nrho, Nlayer_max);     % perfect/reference

nmseC_mat     = nan(Nk, Nrho);
nmseCp_mat    = nan(Nk, Nrho);

%% ====== Sweep kappa0, rho0 ======
for ik = 1:Nk
    kappa0 = kappa0_list(ik);

    for ir = 1:Nrho
        rho0 = rho0_list(ir);

        L = min(floor(kappa0 * Nfeat), Ntrain);
        M = floor((1 - rho0) * Nfeat);
        nnz_count = L * M;

        if L < 1 || M < 1
            fprintf('[SKIP] kappa0=%.4g, rho0=%.4f -> L=%d, M=%d\n', ...
                kappa0, rho0, L, M);
            continue;
        end

        nnz_mat(ik, ir) = nnz_count;
        L_mat(ik, ir)   = L;
        M_mat(ik, ir)   = M;

        fprintf('\n=============================================\n');
        fprintf('kappa0 = %.4g, rho0 = %.4f | L = %d, M = %d, L*M = %d\n', ...
            kappa0, rho0, L, M, nnz_count);

        %% ----- Codebook from source env: C -----
        [Vs, sigmaL, V, info] = psvd_codebook(Htrain, kappa0, rho0, T);

        Z = Htest * Vs;
        decoder = pinv(Vs);
        tildeH = Z * decoder;

        fprintf("NMSE (E', C):\n");
        [~,~,c] = my_print_error(Htest, tildeH);
        nmseC_mat(ik, ir) = c{2};

        %% ----- Codebook from target env: C' -----
        [Vs2, ~, ~, ~] = psvd_codebook(Htrain2, kappa0, rho0, T);

        Z2 = Htest * Vs2;
        decoder2 = pinv(Vs2);
        tildeH2 = Z2 * decoder2;

        fprintf("NMSE (E', C'):\n");
        [~,~,c] = my_print_error(Htest, tildeH2);
        nmseCp_mat(ik, ir) = c{2};

        %% ----- Delay -> Frequency -----
        tildeH4  = reshape(tildeH,  [Ntest, Nt, Nr, Ntap]);
        Htilde   = delay_to_freq_csi(tildeH4, Nsub);

        tildeH42 = reshape(tildeH2, [Ntest, Nt, Nr, Ntap]);
        Htilde2  = delay_to_freq_csi(tildeH42, Nsub);

        %% ----- Rate evaluation -----
        fprintf('Evaluating on SNR0_dB = %f\n', SNR0_dB);

        for Nlayer = 1:Nlayer_max
            fprintf('Nlayer = %d\n', Nlayer);

            Ptot = SNR0 * N0 * Nt * Nr * Nsub * Nlayer / PL;

            fprintf("R(E', C):\n");
            [R_avg, Rperf_avg] = su_mimo_ofdm_rate_imperfect_batch( ...
                Horg_test, H_test, Htilde, Nlayer, Ptot, N0, opts);

            fprintf("R(E', C'):\n");
            [R_avg2, Rperf_avg2] = su_mimo_ofdm_rate_imperfect_batch( ...
                Horg_test, H_test, Htilde2, Nlayer, Ptot, N0, opts);

            RavgC_mat(ik, ir, Nlayer)   = R_avg;
            RavgCp_mat(ik, ir, Nlayer)  = R_avg2;
            RperfC_mat(ik, ir, Nlayer)  = Rperf_avg;
            RperfCp_mat(ik, ir, Nlayer) = Rperf_avg2;

            fprintf('Stored: R(E'',C)=%.4f, R(E'',C'')=%.4f, Rperf=%.4f\n', ...
                R_avg, R_avg2, Rperf_avg2);
        end
    end
end

%% ====== Plot ======
% Choose which Nlayer to visualize
Nlayer_plot = 1;   % change if needed

figure; hold on; grid on; box on;

legend_entries = strings(1, Nk);

% Marker list to rotate
marker_list = {'o','s','d','^','v','>','<','p','h','x','+'};
nMarker = numel(marker_list);

for ik = 1:Nk
    x = squeeze(nnz_mat(ik, :));
    y = squeeze(RavgCp_mat(ik, :, Nlayer_plot));     % plot R(E', C')
    yperf = squeeze(RperfCp_mat(ik, :, Nlayer_plot));

    valid = isfinite(x) & isfinite(y);
    if ~any(valid)
        continue;
    end

    x = x(valid);
    y = y(valid);
    yperf = yperf(valid);

    [x, idx] = sort(x);
    y = y(idx);
    yperf = yperf(idx);

    % rotate marker
    mk = marker_list{mod(ik-1, nMarker) + 1};

    plot(x, y, 'LineWidth', 1.5, ...
        'Marker', mk, 'MarkerSize', 6, ...
        'DisplayName', sprintf('\\kappa_0 = %.4g', kappa0_list(ik)));

    % same kappa0 -> one dotted horizontal reference line
    % yperf_line = mean(yperf, 'omitnan');
    % yline(yperf_line, '--', 'LineWidth', 1.2, 'HandleVisibility', 'off');

    legend_entries(ik) = sprintf('\\kappa_0 = %.4g', kappa0_list(ik));
end

Rmis_global = mean(RavgC_mat(:,:,Nlayer_plot), 'all', 'omitnan');
Rperf_global = mean(RperfCp_mat(:,:,Nlayer_plot), 'all', 'omitnan');
yline(Rmis_global, ':k', 'LineWidth', 1.5, ...
    'DisplayName', 'Mismatched throughput');

yline(Rperf_global, '--k', 'LineWidth', 1.5, ...
    'DisplayName', 'Perfect CSIR/CSIT');

xlabel('Number of nonzero elements');
ylabel('Average spectral efficiency [bps/Hz]');
% title(sprintf('Average spectral efficiency vs. number of nonzero elements (N_{layer} = %d)', Nlayer_plot));
legend('Location', 'best');


%%
save_dir = 'result';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

timestamp = datestr(now, 'yyyymmdd_HHMMSS');
save_file = fullfile(save_dir, sprintf( ...
    'sweep_E%dS%d_to_E%dS%d_SNRcsi%d_SNR0%d_%s.mat', ...
    E, Escenario, Eprime, EprimeScenario, ...
    round(snrCsi_dB), round(SNR0_dB), timestamp));

meta = struct();
meta.snrCsi_dB = snrCsi_dB;
meta.csir_mode = csir_mode;
meta.SNR0_dB = SNR0_dB;
meta.kappa0_list = kappa0_list;
meta.rho0_list = rho0_list;
meta.Ntrain = Ntrain;
meta.Ntest = Ntest;
meta.E = E;
meta.Escenario = Escenario;
meta.Eprime = Eprime;
meta.EprimeScenario = EprimeScenario;
meta.T = T;
meta.Ntot = Ntot;
meta.Nt = Nt;
meta.Nr = Nr;
meta.Nsub = Nsub;
meta.Ntap = Ntap;
meta.Nfeat = Nfeat;
meta.PL = PL;
meta.timestamp = timestamp;

save(save_file, ...
    'meta', ...
    'nnz_mat', 'L_mat', 'M_mat', ...
    'RavgC_mat', 'RavgCp_mat', 'RperfC_mat', 'RperfCp_mat', ...
    'nmseC_mat', 'nmseCp_mat', ...
    '-v7.3');

fprintf('\nSaved results to:\n%s\n', save_file);

fig_file_png = fullfile(save_dir, sprintf( ...
    'throughput_vs_nnz_Nlayer%d_%s.png', Nlayer_plot, timestamp));
fig_file_eps = fullfile(save_dir, sprintf( ...
    'throughput_vs_nnz_Nlayer%d_%s.eps', Nlayer_plot, timestamp));

saveas(gcf, fig_file_png);
print(gcf, fig_file_eps, '-depsc');

fprintf('Saved figure to:\n%s\n%s\n', fig_file_png, fig_file_eps);