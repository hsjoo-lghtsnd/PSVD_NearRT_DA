clc; clear;

%% ====== USER SETTINGS ======
snrCsi_dB = 20;               % CSI-RS SNR definition used for Horg -> H
csir_mode = 'global';         % 'global' | 'perSample' | 'perSampleSub'

SNR0_dB_list = [0 5 10 15 20 25 30];   % <-- sweep list

kappa0 = 0.01;
rho0 = 0.9;

Ntrain = 100;
Ntest = 1000;

E=2; Escenario=5;
Eprime=2; EprimeScenario=1;

%% ====== (1) Horg -> H (imperfect CSIR), Transform to time-domain ======

Ntot = Ntrain+Ntest;

E_opts = struct;
E_opts.seed = 12345;
[Ht, Horg] = load_environment(E, Ntot, Escenario, E_opts);

[H, sigma2_csi] = simulate_imperfect_csir_batch(Horg, snrCsi_dB, csir_mode);
if (E==2)
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
    if (Eprime==2)
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

%% ====== (2) PSVD codebook construction ======
T = 30;

[Vs, sigmaL, V, info] = psvd_codebook(Htrain, kappa0, rho0, T);

Z = Htest * Vs;
decoder = pinv(Vs);
tildeH = Z * decoder;

fprintf("NMSE (E', C):\n");
my_print_error(Htest, tildeH);

[Vs2, ~, ~, ~] = psvd_codebook(Htrain2, kappa0, rho0, T);

Z2 = Htest * Vs2;
decoder2 = pinv(Vs2);
tildeH2 = Z2 * decoder2;

fprintf("NMSE (E', C'):\n");
my_print_error(Htest, tildeH2);

%% ====== (3) Prepare frequency-domain channels ======
Nt = size(Horg, 2);
Nr = size(Horg, 3);
Nlayer_max = max(min(Nt,Nr),1);
N0   = 1e-2;                  % AWGN variance used in data link evaluation
Nsub = size(Horg,4);

PL = mean(abs(Horg(:)).^2);   % channel gain (path loss)

Horg_test = Horgprime(Ntrain+1:Ntrain+Ntest,:,:,:);
H_test    = Hprime(Ntrain+1:Ntrain+Ntest,:,:,:);

Ntap = size(Ht,4);

tildeH4 = reshape(tildeH,  [Ntest, Nt, Nr, Ntap]);
Htilde  = delay_to_freq_csi(tildeH4, Nsub);

tildeH42 = reshape(tildeH2, [Ntest, Nt, Nr, Ntap]);
Htilde2  = delay_to_freq_csi(tildeH42, Nsub);

%% ====== (4) Sweep over SNR0 ======
Ns = numel(SNR0_dB_list);

R_mismatch = nan(Nlayer_max, Ns);   % R(E', C)
R_adapted  = nan(Nlayer_max, Ns);   % R(E', C')
R_perf     = nan(Nlayer_max, Ns);   % perfect CSIR/CSIT

opts = struct('use_pinv', true, 'reg_epsilon', 1e-6, 'return_cells', false);

for iSNR = 1:Ns
    SNR0_dB = SNR0_dB_list(iSNR);
    SNR0    = 10^(SNR0_dB/10);

    fprintf('\n=============================================\n');
    fprintf('Evaluating on SNR0_dB = %.2f\n', SNR0_dB);

    for Nlayer = 1:Nlayer_max
        fprintf('Nlayer = %d\n', Nlayer);

        Ptot = SNR0 * N0 * Nt * Nr * Nsub * Nlayer / PL;

        fprintf("R(E',C):\n");
        [R_avg, Rperf_avg] = su_mimo_ofdm_rate_imperfect_batch( ...
            Horg_test, H_test, Htilde, Nlayer, Ptot, N0, opts);

        fprintf("R(E',C'):\n");
        [R_avg2, Rperf_avg2] = su_mimo_ofdm_rate_imperfect_batch( ...
            Horg_test, H_test, Htilde2, Nlayer, Ptot, N0, opts);

        R_mismatch(Nlayer, iSNR) = R_avg;
        R_adapted(Nlayer,  iSNR) = R_avg2;
        R_perf(Nlayer,     iSNR) = Rperf_avg2;   % should be same as Rperf_avg
    end
end

%% ====== (5) Plot: one figure per Nlayer ======
for Nlayer = 1:Nlayer_max
    figure; hold on; grid on; box on;

    plot(SNR0_dB_list, R_mismatch(Nlayer,:), '-o', ...
        'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Mismatched $(E'',C)$');

    plot(SNR0_dB_list, R_adapted(Nlayer,:), '-s', ...
        'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', 'Adapted $(E'',C'')$');

    plot(SNR0_dB_list, R_perf(Nlayer,:), '--', ...
        'LineWidth', 1.8, ...
        'DisplayName', 'Perfect CSIR/CSIT');

    xlabel('SNR_0 [dB]');
    ylabel('Average spectral efficiency [bps/Hz]');
    title(sprintf('N_{layer} = %d', Nlayer), 'Interpreter', 'tex');

    legend('Location', 'best');
end