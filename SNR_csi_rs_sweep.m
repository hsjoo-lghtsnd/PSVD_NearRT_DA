clc; clear;

%% ====== USER SETTINGS ======
csir_mode = 'global';         % 'global' | 'perSample' | 'perSampleSub'
snrCsi_list_dB = [0 5 10 15 20 25 30 40];

SNR0_dB = 20;                 % fixed data-link SNR
Ntrain = 0;
Ntest  = 1000;

% Representative environment
E = 2;            % CDL
Escenario = 1;    % CDL-A

%% ====== LOAD TRUE CHANNEL ======
Ntot = Ntrain + Ntest;

E_opts = struct;
E_opts.seed = 12345;

[Ht, Horg] = load_environment(E, Ntot, Escenario, E_opts);

fprintf('Environment load completed: E%d, scenario %d\n', E, Escenario);

Nt   = size(Horg, 2);
Nr   = size(Horg, 3);
Nsub = size(Horg, 4);

Nlayer_max = min(Nt, Nr);

fprintf('Detected dimensions: Nt=%d, Nr=%d, Nsub=%d\n', Nt, Nr, Nsub);

%% ====== COMMON PARAMETERS ======
SNR0 = 10^(SNR0_dB/10);
N0   = 1e-2;

PL = mean(abs(Horg(:)).^2);   % path loss proxy

Horg_test = Horg(Ntrain+1:Ntrain+Ntest,:,:,:);

opts = struct('use_pinv', true, 'reg_epsilon', 1e-6, 'return_cells', false);

Ns = numel(snrCsi_list_dB);

Ravg_mat   = nan(Ns, Nlayer_max);
Rperf_mat  = nan(Ns, Nlayer_max);
nmse_mat   = nan(Ns, 1);

%% ====== SWEEP OVER CSI-RS SNR ======
for is = 1:Ns
    snrCsi_dB = snrCsi_list_dB(is);

    fprintf('\n=============================================\n');
    fprintf('SNR_CSI-RS = %.1f dB\n', snrCsi_dB);

    % Imperfect CSIR at UE
    [H, sigma2_csi] = simulate_imperfect_csir_batch(Horg, snrCsi_dB, csir_mode);

    H_test = H(Ntrain+1:Ntrain+Ntest,:,:,:);

    % Perfect feedback assumption:
    % BS receives the UE-side imperfect estimate without additional loss
    Htilde = H_test;

    % Optional diagnostic NMSE between true channel and imperfect CSIR
    num = sum(abs(Horg_test(:) - H_test(:)).^2);
    den = sum(abs(Horg_test(:)).^2);
    nmse_mat(is) = 10*log10(num / den);

    fprintf('CSIR NMSE = %.4f dB\n', nmse_mat(is));

    for Nlayer = 1:Nlayer_max
        fprintf('Nlayer = %d\n', Nlayer);

        % Keep the same normalization as the main text:
        % total power scales with Nlayer
        Ptot = SNR0 * N0 * Nt * Nr * Nsub * Nlayer / PL;

        [R_avg, Rperf_avg] = su_mimo_ofdm_rate_imperfect_batch( ...
            Horg_test, H_test, Htilde, Nlayer, Ptot, N0, opts);

        Ravg_mat(is, Nlayer)  = R_avg;
        Rperf_mat(is, Nlayer) = Rperf_avg;

        fprintf('Average R = %.4f bps/Hz | Perfect ref = %.4f bps/Hz\n', ...
            R_avg, Rperf_avg);
    end
end

%% ====== PLOT 1: throughput vs Nlayer ======
figure; hold on; grid on; box on;

marker_list = {'o','s','d','^','v','>','<','p','h','x','+'};
nMarker = numel(marker_list);
legend_entries = strings(1, Ns);

for is = 1:Ns
    x = 1:Nlayer_max;
    y = Ravg_mat(is, :);

    mk = marker_list{mod(is-1, nMarker) + 1};

    plot(x, y, 'LineWidth', 1.5, ...
        'Marker', mk, 'MarkerSize', 6, ...
        'DisplayName', sprintf('SNR_{CSI-RS} = %.0f dB', snrCsi_list_dB(is)));
end

% Global perfect-reference line
Rperf_curve = mean(Rperf_mat, 1, 'omitnan');   % 1 x Nlayer_max
plot(1:Nlayer_max, Rperf_curve, '--k', 'LineWidth', 1.8, ...
    'DisplayName', 'Perfect CSIR/CSIT');

xticks(1:Nlayer_max);
xlabel('Number of layers');
ylabel('Average spectral efficiency [bps/Hz]');
% title('Layer-wise throughput under imperfect CSIR with perfect feedback');
legend('Location','eastoutside');

%% ====== PLOT 2: throughput vs SNR_CSI-RS ======
figure; hold on; grid on; box on;

legend_entries2 = strings(1, Nlayer_max);

for Nlayer = 1:Nlayer_max
    x = snrCsi_list_dB;
    y = Ravg_mat(:, Nlayer).';

    mk = marker_list{mod(Nlayer-1, nMarker) + 1};

    plot(x, y, 'LineWidth', 1.5, ...
        'Marker', mk, 'MarkerSize', 6, ...
        'DisplayName', sprintf('N_{layer} = %d', Nlayer));
end

yline(Rperf_global, '--k', 'LineWidth', 1.5, ...
    'DisplayName', 'Perfect CSIR/CSIT');

xlabel('SNR_{CSI-RS} [dB]');
ylabel('Average spectral efficiency [bps/Hz]');
title('Impact of CSI-RS quality on layer-wise throughput');
legend('Location', 'best');

%% ====== OPTIONAL: plot CSIR NMSE ======
figure; hold on; grid on; box on;
plot(snrCsi_list_dB, nmse_mat, '-o', 'LineWidth', 1.5, 'MarkerSize', 6);
xlabel('SNR_{CSI-RS} [dB]');
ylabel('CSIR NMSE [dB]');
title('Channel estimation quality vs CSI-RS SNR');