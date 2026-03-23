%% ====== USER SETTINGS ======
snrCsi_dB = 20;               % CSI-RS SNR definition used for Horg -> H
csir_mode = 'global';         % 'global' | 'perSample' | 'perSampleSub'

SNR0_dB = 20;

Ntot = 2000;
Ntrain = 100;
Ntest = 1000;

kappa0 = 0.02;
rho0 = 0.8;

%% (0) CSI Data Generation

E=4;

% E2
cdlProfile = 'CDL-A';
delaySpread = 300e-9;
seed = 20260226;

% E4
E4scenario_choice = 4;
E4_scenarios = {...
    'Indoor_CloselySpacedUser_2_6GHz'; ...        % : ~3 taps
    'IndoorHall_5GHz'; ...                        % : ~10 taps
    'SemiUrban_CloselySpacedUser_2_6GHz'; ...     % : ~50 taps
    'SemiUrban_300MHz'; ...                       % : ~167 taps
    'SemiUrban_VLA_2_6GHz' ...                    % : ~167 taps
    };


if (E==2)
    
Horg = CDL_create_wrapper(Ntot, cdlProfile, delaySpread, seed);
end

if (E==4)
addpath('COST2100_MATLAB');

Nt=32;
Nr=4;
Ntap_gen=250;
E4scenario = E4_scenarios{E4scenario_choice};

opts = struct();
opts.seed = 1;
opts.linkMode = 'auto';   % or 'Single', 'Multiple'
opts.verbose = true;

[Hset, ~, ~] = generate_cost2100_dataset( ...
    E4scenario, ...
    'LOS', ...
    Nt, ...      % Nt
    Nr, ...      % Nr
    Ntap_gen, ...     % Ntap
    Ntot, ...    % Ntrain
    opts);

Horg = delay_to_freq_csi(Hset, 624);
end


%% ====== (1) Horg -> H (imperfect CSIR) ======
[H, sigma2_csi] = simulate_imperfect_csir_batch(Horg, snrCsi_dB, csir_mode);

%% ===== (2) Transform to time-domain =====

if (E==2)
    Ntap = 32;
    Ht = CDL_convert_wrapper(H, Ntap);
else
    Ntap = 32;
    Ht = freq_to_delay_csi(H, Ntap);
end

Htrain = reshape(Ht(1:Ntrain,:,:,:), Ntrain, []);
Htest = reshape(Ht(Ntrain+1:Ntrain+Ntest, :, :, :), Ntest, []);

%%
BB = [200, 500, 1000, 2000, 3000];
T = 30;
P = 0:0.1:2;

p = 0;
[tildeH, ~, ~, ~] = psvd_Htrain_wrapper(Htrain, Htest, kappa0, rho0, T, p, Btot);
[~,~,c] = my_print_error(Htest, tildeH);
pivot_NMSE = c{2};

NMSE = zeros(length(BB), length(P));
Bits = zeros(length(BB), length(P));
for i1=1:length(BB)
    Btot = BB(i1);
    

    for i2=1:length(P)
        p = P(i2);
        disp(p)
        [~, QtildeH, ~, rBtot] = psvd_Htrain_wrapper(Htrain, Htest, kappa0, rho0, T, p, Btot);
        disp(rBtot)
        Bits(i1,i2) = rBtot;

        [~,~,c2] = my_print_error(Htest, QtildeH);
        NMSE(i1,i2) = c2{2};
    end
end

%%
figure;
hold on;
grid minor;
box on;

markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', 'x', '+'};

for i = 1:length(BB)
    mk = markers{mod(i-1, length(markers)) + 1};
    plot(P, NMSE(i,:), ...
        'LineWidth', 1.5, ...
        'Marker', mk, ...
        'MarkerSize', 7, ...
        'MarkerIndices', 1:2:length(P));
end

plot([P(1), P(end)], [pivot_NMSE, pivot_NMSE], '--k', 'LineWidth', 1.5);

legend('B_{tot}=200', 'B_{tot}=500', 'B_{tot}=1000', 'B_{tot}=2000',...
    'B_{tot}=3000', 'No Quantization');
xlabel('Hyperparameter p');
ylabel('Average NMSE (dB)');

set(gca,'FontName','Times New Roman',...
        'FontSize',16,...
        'LineWidth',1.2)

set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',16)

%%
figure;
hold on;
grid minor;
box on;

markers = {'o', 's', '^', 'd', 'v', '>', '<', 'p', 'h', 'x', '+'};

for i = 1:length(BB)
    mk = markers{mod(i-1, length(markers)) + 1};
    plot(P, Bits(i,:), ...
        'LineWidth', 1.5, ...
        'Marker', mk, ...
        'MarkerSize', 7, ...
        'MarkerIndices', 1:2:length(P));
end

legend('B_{tot}=200', 'B_{tot}=500', 'B_{tot}=1000', 'B_{tot}=2000',...
    'B_{tot}=3000');
xlabel('Hyperparameter p');
ylabel('Realized Bits per Codeword');
ylim([0, 3000]);

set(gca,'FontName','Times New Roman',...
        'FontSize',16,...
        'LineWidth',1.2)

set(findall(gcf,'-property','FontName'),'FontName','Times New Roman')
set(findall(gcf,'-property','FontSize'),'FontSize',16)

%% ====== (3) Evaluate R over selected samples ======

SNR0 = 10^(SNR0_dB/10);

Nsym = max(min(Nt,Nr)-3,1);
N0   = 1e-2;                  % AWGN variance used in data link evaluation
Nsub = size(Horg,4);

PL = mean(abs(Horg(:)).^2);   % channel gain

Ptot = SNR0*N0*Nt*Nr*Nsub*Nsym/PL;

Neval = 200;
% Example: random subset
idxEval = randperm(Ntest, Neval);

opts = struct('use_pinv', true, 'reg_epsilon', 1e-6, 'return_cells', false);

tildeH4 = reshape(tildeH, [Ntest, size(H,2), size(H,3), Ntap]);
Htilde = delay_to_freq_csi(tildeH4,size(H,4));

Horg_test = Horg(Ntrain+1:Ntrain+Ntest,:,:,:);
H_test = H(Ntrain+1:Ntrain+Ntest,:,:,:);

R_list = zeros(Neval,1);
Rperf_list = zeros(Neval,1);
for ii = 1:Neval
    t = idxEval(ii);

    Horg_t   = squeeze(Horg_test(t,:,:,:));    % [Nt,Nr,Nsub]
    H_t      = squeeze(H_test(t,:,:,:));       % [Nt,Nr,Nsub]
    Htilde_t = squeeze(Htilde(t,:,:,:));  % [Nt,Nr,Nsub]

    [R_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, H_t, Htilde_t, Nsym, Ptot, N0);
    [Rperf_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, Horg_t, Horg_t, Nsym, Ptot, N0);
end

R_avg = mean(R_list);
Rperf_avg = mean(Rperf_list);
fprintf("Average R over %d samples = %.6f bps/Hz\n", Neval, R_avg);
fprintf("Average R over %d samples = %.6f bps/Hz (when perfect CSIR/CSIT)\n", Neval, Rperf_avg);

