function [Ht, Horg] = load_environment(E, Ntot, scenario_choice, E_opts)
% returns Ht: time-domain. Horg: frequency-domain original.

% E1
E1_scenarios = {...
    "spot1_3.5G_bus.mat"
    "spot1_3.5G_no_bus.mat"
    "spot2_3.5G_no_bus.mat"
    "spot3_3.5G_no_bus.mat"
    };

% E2
E2_scenarios = {...
    'CDL-A'
    'CDL-B'
    'CDL-C'
    'CDL-D'
    'CDL-E'
    };
cdlProfile = E2_scenarios{scenario_choice};

% E3
E3_scenarios = {
    "indoor"
    "outdoor"
    };

% E4
E4_scenarios = {...
    'Indoor_CloselySpacedUser_2_6GHz'; ...        % : ~3 taps
    'IndoorHall_5GHz'; ...                        % : ~10 taps
    'SemiUrban_CloselySpacedUser_2_6GHz'; ...     % : ~50 taps
    'SemiUrban_300MHz'; ...                       % : ~167 taps
    'SemiUrban_VLA_2_6GHz' ...                    % : ~167 taps
    };

% E2
delaySpread = 300e-9;
seed = 20260226;
Nsub = 624; % 52 x 12
SCS = 15;                      % kHz
TxArraySize = [4 4 2 1 1];     % [rows, cols, pol, panelRows, panelCols]
RxArraySize = [2 2 1 1 1];
Ntap = 32;

% E3
maxDelay = 32;
nTx = 32;
numChannels = 2;

% E4
Nt=32;
Nr=4;
maxRetry = 200;

% Optional input
if nargin >= 4 && ~isempty(E_opts)
    if isfield(E_opts, 'delaySpread') && ~isempty(E_opts.delaySpread)
        delaySpread = E_opts.delaySpread;
        if E==2
            fprintf('Specified delay spread of %e\n', delaySpread);
        end
    end
    if isfield(E_opts, 'seed') && ~isempty(E_opts.seed)
        seed = E_opts.seed;
    end
    if isfield(E_opts, 'Nsub') && ~isempty(E_opts.Nsub)
        Nsub = E_opts.Nsub;
    end
    if isfield(E_opts, 'SCS') && ~isempty(E_opts.SCS)
        SCS = E_opts.SCS;
    end
    if isfield(E_opts, 'TxArraySize') && ~isempty(E_opts.TxArraySize)
        TxArraySize = E_opts.TxArraySize;
    end
    if isfield(E_opts, 'RxArraySize') && ~isempty(E_opts.RxArraySize)
        RxArraySize = E_opts.RxArraySize;
    end
    if isfield(E_opts, 'Ntap') && ~isempty(E_opts.Ntap)
        Ntap = E_opts.Ntap;
    end
    if isfield(E_opts, 'Nt') && ~isempty(E_opts.Nt)
        Nt = E_opts.Nt;
    end
    if isfield(E_opts, 'Nr') && ~isempty(E_opts.Nr)
        Nr = E_opts.Nr;
    end
    if isfield(E_opts, 'maxRetry') && ~isempty(E_opts.maxRetry)
        maxRetry = E_opts.maxRetry;
    end
end

if E==1
    load(fullfile("data",E1_scenarios{scenario_choice}),'H_bus');
    sampleSize = size(H_bus,1);
    Ht = reshape(H_bus, sampleSize, 100, 1, 100);

    rng(seed);
    idx = randperm(sampleSize, Ntot);
    Ht = Ht(idx,:,:,:);

elseif E==2
    % Original frequency CSI
    Horg = generate_cdl_freq_csi(Ntot, SCS, Nsub, TxArraySize, RxArraySize, delaySpread, cdlProfile, seed);
    
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
    [Hobs, ~, ~] = extract_csirs_observations(Horg, carrier, csirs);
    
    Ht = freq_to_delay_csi(Hobs, Ntap); % [Ns, Nt, Nr, Ntap]
    disp(size(Ht));
    
elseif E==3
    environment = E3_scenarios{scenario_choice};
    
    % Load test data
    load(fullfile("data","DATA_Htest"+extractBefore(environment,"door")+".mat"), 'HT');
    sampleSize = length(HT);
    xTest = reshape(HT', maxDelay, nTx, numChannels, sampleSize);
    xTest = permute(xTest, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

    rng(seed);
    idx = randperm(sampleSize, Ntot);
    
    xt = permute(xTest(:,:,:,idx), [4, 1, 3, 2])-0.5; % inverse permutation of [2,1,3,4]
    Ht = zeros(Ntot, 32, 1, 32);
    Ht(:,:,1,:) = xt(:,:,1,:) + 1j*xt(:,:,2,:);
elseif E==4
    addpath('COST2100_MATLAB');

    Ntap_gen=250;
    E4scenario = E4_scenarios{scenario_choice};
    
    opts = struct();
    opts.seed = seed;
    opts.linkMode = 'auto';   % or 'Single', 'Multiple'
    opts.verbose = true;
    opts.maxRetry = maxRetry;
    
    [Hset, ~, ~] = generate_cost2100_dataset( ...
        E4scenario, ...
        'LOS', ...
        Nt, ...      % Nt
        max(Nr, 2), ...      % Nr
        Ntap_gen, ...     % Ntap
        Ntot, ...    % Ntot
        opts);
    
    Hset = Hset(:,:,1:Nr,:);
    Horg = delay_to_freq_csi(Hset, Nsub);
    Ht = freq_to_delay_csi(Horg, Ntap);
elseif E==5
    % Load test data
    load(fullfile("data","DATA_Htestrandom.mat"), 'HT');
    sampleSize = length(HT);
    xTest = reshape(HT', maxDelay, nTx, numChannels, sampleSize);
    xTest = permute(xTest, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

    rng(seed);
    idx = randperm(sampleSize, Ntot);
    
    xt = permute(xTest(:,:,:,idx), [4, 1, 3, 2])-0.5; % inverse permutation of [2,1,3,4]
    Ht = zeros(Ntot, 32, 1, 32);
    Ht(:,:,1,:) = xt(:,:,1,:) + 1j*xt(:,:,2,:);
 end

if (~exist('Horg','var'))
    Horg = delay_to_freq_csi(Ht, Nsub);
end


end
