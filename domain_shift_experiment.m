clc; clear;
% % E1
% E1_scenarios = {...
%     "spot1_3.5G_bus.mat"
%     "spot1_3.5G_no_bus.mat"
%     "spot2_3.5G_no_bus.mat"
%     "spot3_3.5G_no_bus.mat"
%     };
% 
% % E2
% cdlProfile = 'CDL-A'; % A<-1, B<-2, ..., E<-5
% delaySpread = 300e-9;
% seed = 20260226;
% 
% % E3
% E3_scenarios = {
%     "indoor"
%     "outdoor"
%     };
% 
% % E4
% E4_scenarios = {...
%     'Indoor_CloselySpacedUser_2_6GHz'; ...        % : ~3 taps
%     'IndoorHall_5GHz'; ...                        % : ~10 taps
%     'SemiUrban_CloselySpacedUser_2_6GHz'; ...     % : ~50 taps
%     'SemiUrban_300MHz'; ...                       % : ~167 taps
%     'SemiUrban_VLA_2_6GHz' ...                    % : ~167 taps
%     };
%% ====== USER SETTINGS ======
snrCsi_dB = 20;               % CSI-RS SNR definition used for Horg -> H
csir_mode = 'global';         % 'global' | 'perSample' | 'perSampleSub'

SNR0_dB = 20;

rho0 = 0.8;

MCcount = 1000;
Nth = 3;

%% Data loading (Environment Selection)

E=4;
scenario_choice = 5;
scenario_choice_2 = 1;

E_opts = struct;
E_opts.seed = 20260226;
maxXi = zeros(MCcount,1);
maxProxy = zeros(MCcount,1);

T = 30;
p = 0.5;
Btot = 500;

for i=1:MCcount
    disp(i);
    % Htrain generation on E
    E_opts.delaySpread = 300e-9;
    Ns=100;
    [Ht, Horg] = load_environment(E, Ns, scenario_choice, E_opts);
    [H, ~] = simulate_imperfect_csir_batch(Horg, snrCsi_dB, csir_mode);
    Nt = size(Ht,2);
    Nr = size(Ht,3);
    Ntap = size(Ht,4);
    if (E==2)
        Htrain = CDL_convert_wrapper(H, Ntap);
    else
        Htrain = freq_to_delay_csi(H, Ntap);
    end
    
    E_opts.seed = E_opts.seed+1;

    % Htest generation on E'
    Ns=Nth;
    E_opts.delaySpread = 30e-9;
    [~, Horg] = load_environment(E, Ns, scenario_choice_2, E_opts);
    [H, sigma2_csi] = simulate_imperfect_csir_batch(Horg, snrCsi_dB, csir_mode);
    if (E==2)
        Htest = CDL_convert_wrapper(H, Ntap);
    else
        Htest = freq_to_delay_csi(H, Ntap);
    end
    
    E_opts.seed = E_opts.seed+1;
    
    Ntrain = size(Htrain,1);
    Ntest = size(Htest,1);
    htrain = reshape(Htrain, Ntrain, []);
    htest = reshape(Htest, Ntest, []);
    
    % PSVD
    kappa0 = 50.5/(Nt*Nr*Ntap); % L = 50

    [Vs, sigmaL, ~, ~] = psvd_codebook(htrain, kappa0, rho0, T);
    decoder = pinv(Vs);
    
    Z = htest*Vs;
    sigmaL = sigmaL/sqrt(Ntrain);
    bSeq = bit_alloc(sigmaL, Btot, p);
    
    Zq = quantize_wrapper(Z, bSeq, sigmaL);
    
    QtildeH = Zq*decoder;

    proxy = 1 - (sum(abs(Zq).^2, 2) ./ sum(abs(htest).^2, 2));
    
    % calculate xi (true NMSE) and store
    [~, ~, container] = my_print_error(htest, QtildeH);
    
    disp(container{3});
    maxXi(i) = container{3};

    maxProxy(i) = 10*log10(max(proxy));
    disp(maxProxy(i));
end
disp(10*log10(mean(10.^(maxProxy/10))));
disp(10*log10(mean(10.^(maxXi/10))));


%% ====== (3) Evaluate R over selected samples ======

% SNR0 = 10^(SNR0_dB/10);
% 
% Nsym = max(min(Nt,Nr)-3,1);
% N0   = 1e-2;                  % AWGN variance used in data link evaluation
% Nsub = size(Horg,4);
% 
% PL = mean(abs(Horg(:)).^2);   % channel gain
% 
% Ptot = SNR0*N0*Nt*Nr*Nsub*Nsym/PL;
% 
% Neval = 200;
% % Example: random subset
% idxEval = randperm(Ntest, Neval);
% 
% opts = struct('use_pinv', true, 'reg_epsilon', 1e-6, 'return_cells', false);
% 
% tildeH4 = reshape(tildeH, [Ntest, size(H,2), size(H,3), Ntap]);
% Htilde = delay_to_freq_csi(tildeH4,size(H,4));
% 
% Horg_test = Horg(Ntrain+1:Ntrain+Ntest,:,:,:);
% H_test = H(Ntrain+1:Ntrain+Ntest,:,:,:);
% 
% R_list = zeros(Neval,1);
% Rperf_list = zeros(Neval,1);
% for ii = 1:Neval
%     t = idxEval(ii);
% 
%     Horg_t   = reshape(Horg_test(t,:,:,:), [Nt, Nr, Nsub]);    % [Nt,Nr,Nsub]
%     H_t      = reshape(H_test(t,:,:,:), [Nt, Nr, Nsub]);       % [Nt,Nr,Nsub]
%     Htilde_t = reshape(Htilde(t,:,:,:), [Nt, Nr, Nsub]);  % [Nt,Nr,Nsub]
% 
%     [R_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, H_t, Htilde_t, Nsym, Ptot, N0);
%     [Rperf_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, Horg_t, Horg_t, Nsym, Ptot, N0);
% end
% 
% R_avg = mean(R_list);
% Rperf_avg = mean(Rperf_list);
% fprintf("Average R over %d samples = %.6f bps/Hz\n", Neval, R_avg);
% fprintf("Average R over %d samples = %.6f bps/Hz (when perfect CSIR/CSIT)\n", Neval, Rperf_avg);

