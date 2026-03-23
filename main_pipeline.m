clc; clear;

%% ====== USER SETTINGS ======
snrCsi_dB = 20;               % CSI-RS SNR definition used for Horg -> H
csir_mode = 'global';         % 'global' | 'perSample' | 'perSampleSub'

SNR0_dB = 20;

kappa0 = 0.01;
rho0 = 0.9;

Ntrain = 100;
Ntest = 1000;

E=4; Escenario=1;
Eprime=4; EprimeScenario=3;



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
    if (E==2)
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
Htest = reshape(Htprime(Ntrain+1:Ntrain+Ntest,:,:,:), Ntest, []);


%%
T = 30;

[Vs, sigmaL, V, info] = psvd_codebook(Htrain, kappa0, rho0, T);

Z = Htest*Vs;

decoder = pinv(Vs);
tildeH = Z*decoder;

my_print_error(Htest, tildeH);

%%%%

[Vs2, ~, ~, ~] = psvd_codebook(Htrain2, kappa0, rho0, T);

Z2 = Htest*Vs2;

decoder2 = pinv(Vs2);
tildeH2 = Z2*decoder2;

my_print_error(Htest, tildeH2);


%% ====== (3) Evaluate R over selected samples ======

SNR0 = 10^(SNR0_dB/10);
Nt = size(Horg, 2);
Nr = size(Horg, 3);
Nlayer_max = max(min(Nt,Nr),1);
N0   = 1e-2;                  % AWGN variance used in data link evaluation
Nsub = size(Horg,4);

PL = mean(abs(Horg(:)).^2);   % channel gain (path loss)

Horg_test = Horgprime(Ntrain+1:Ntrain+Ntest,:,:,:);
H_test = Hprime(Ntrain+1:Ntrain+Ntest,:,:,:);

Ntap = size(Ht,4);
tildeH4 = reshape(tildeH, [Ntest, Nt, Nr, Ntap]);
Htilde = delay_to_freq_csi(tildeH4, Nsub);

tildeH42 = reshape(tildeH2, [Ntest, Nt, Nr, Ntap]);
Htilde2 = delay_to_freq_csi(tildeH42, Nsub);

opts = struct('use_pinv', true, 'reg_epsilon', 1e-6, 'return_cells', false);
for Nlayer=1:Nlayer_max
    fprintf('Nlayer = %d\n', Nlayer);
    Ptot = SNR0*N0*Nt*Nr*Nsub*Nlayer/PL;
    fprintf("R(E',C):\n");
    [R_avg, Rperf_avg] = su_mimo_ofdm_rate_imperfect_batch(Horg_test, H_test, Htilde, Nlayer, Ptot, N0, opts);

    fprintf("R(E',C'):\n");
    [R_avg2, Rperf_avg2] = su_mimo_ofdm_rate_imperfect_batch(Horg_test, H_test, Htilde2, Nlayer, Ptot, N0, opts);
end
