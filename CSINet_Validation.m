% This MATLAB script trains CSINet model proposed in
% 'Chao-Kai Wen, Wan-Ting Shih, and Shi Jin, "Deep learning for massive MIMO CSI feedback,”
% IEEE Wireless Communications Letters, 2018. [Online]. Available: https://ieeexplore.ieee.org/document/8322184/.'
% using in MATLAB®.

clc; clear;

% Set network parameters
maxDelay = 32;
nTx = 32;
numChannels = 2;
compressRate = 1/64; % 1/4 | 1/16 | 1/32 | 1/64

% Create CSINet deep network
CSINet = createCSINet(maxDelay, nTx, numChannels, compressRate);

% % Analyze CSINet architecture visually
% analyzeNetwork(CSINet);

%% Data loading (Environment Selection)
Ns = 2000;
Ntrain = 100;
Nval = 500;
Ntest = 1000;

E=5


Nt = nTx;
Nr = 1;
Ntap = maxDelay;
Nsub = 624;

if E==1
    load(fullfile("data","spot1_3.5G_bus.mat"));
    Ht = reshape(H_bus, 5000, 100, 1, 100);

    idx = randperm(5000, Ns);
    Ht = Ht(idx,1:nTx,:,1:maxDelay);
    
    Ht_real = real(Ht);
    Ht_imag = imag(Ht);
    
    globalMin = min([Ht_real(:); Ht_imag(:)]);
    globalMax = max([Ht_real(:); Ht_imag(:)]);
    
    scale = globalMax - globalMin;
    
    xTrain = zeros(nTx, maxDelay, numChannels, Ntrain); % [Nt, Ntap, Re/Im, Ns]
    xTrain(:,:,1,:) = real(permute(Ht(Nval+1:Nval+Ntrain, :, :, :), [2, 4, 3, 1]));
    xTrain(:,:,2,:) = imag(permute(Ht(Nval+1:Nval+Ntrain, :, :, :), [2, 4, 3, 1]));

    xVal = zeros(nTx, maxDelay, numChannels, Nval); % [Nt, Ntap, Re/Im, Ns]
    xVal(:,:,1,:) = real(permute(Ht(1:Nval, :, :, :), [2, 4, 3, 1]));
    xVal(:,:,2,:) = imag(permute(Ht(1:Nval, :, :, :), [2, 4, 3, 1]));
    
    xTest = zeros(nTx, maxDelay, numChannels, Ntest);
    xTest(:,:,1,:) = real(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));
    xTest(:,:,2,:) = imag(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));
    
    xTrainNorm = (xTrain - globalMin) / scale;
    xValNorm   = (xVal   - globalMin) / scale;
    xTestNorm  = (xTest  - globalMin) / scale;

elseif E==2
    SCS = 15;                      % kHz
    TxArraySize = [4 4 2 1 1];     % [rows, cols, pol, panelRows, panelCols]
    RxArraySize = [1 1 1 1 1];
    delaySpread = 300e-9;          % seconds
    cdlProfile = "CDL-B";
    seed = 20260226;

    Ntap = 32;

    % Original frequency CSI
    Horg = generate_cdl_freq_csi(Ns, SCS, Nsub, TxArraySize, RxArraySize, delaySpread, cdlProfile, seed);
    
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
    [Hobs, scIdx, maskInfo] = extract_csirs_observations(Horg, carrier, csirs);
    
    Ht = freq_to_delay_csi(Hobs, Ntap); % [Ns, Nt, Nr, Ntap]
    disp(size(Ht));
    
    Ht_real = real(Ht);
    Ht_imag = imag(Ht);
    
    globalMin = min([Ht_real(:); Ht_imag(:)]);
    globalMax = max([Ht_real(:); Ht_imag(:)]);
    
    scale = globalMax - globalMin;
    
    xTrain = zeros(nTx, maxDelay, numChannels, Ntrain); % [Nt, Ntap, Re/Im, Ns]
    xTrain(:,:,1,:) = real(permute(Ht(Nval+1:Nval+Ntrain, :, :, :), [2, 4, 3, 1]));
    xTrain(:,:,2,:) = imag(permute(Ht(Nval+1:Nval+Ntrain, :, :, :), [2, 4, 3, 1]));

    xVal = zeros(nTx, maxDelay, numChannels, Nval); % [Nt, Ntap, Re/Im, Ns]
    xVal(:,:,1,:) = real(permute(Ht(1:Nval, :, :, :), [2, 4, 3, 1]));
    xVal(:,:,2,:) = imag(permute(Ht(1:Nval, :, :, :), [2, 4, 3, 1]));
    
    xTest = zeros(nTx, maxDelay, numChannels, Ntest);
    xTest(:,:,1,:) = real(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));
    xTest(:,:,2,:) = imag(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));
    
    xTrainNorm = (xTrain - globalMin) / scale;
    xValNorm   = (xVal   - globalMin) / scale;
    xTestNorm  = (xTest  - globalMin) / scale;
elseif E==3

    environment = "indoor"; % "indoor" | "outdoor"
    
    % Load test data
    load(fullfile("data","DATA_Htest"+extractBefore(environment,"door")+".mat"));
    sampleSize = length(HT);
    xTest = reshape(HT', maxDelay, nTx, numChannels, sampleSize);
    xTest = permute(xTest, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

    idx = randperm(sampleSize, Ns);
    
    xt = permute(xTest(:,:,:,idx), [4, 1, 3, 2])-0.5; % inverse permutation of [2,1,3,4]
    Ht = zeros(Ns, Nt, 1, Ntap);
    Ht(:,:,1,:) = xt(:,:,1,:) + 1j*xt(:,:,2,:);

    xTrainNorm = xTest(:,:,:,idx(Nval+1:Nval+Ntrain));
    xValNorm = xTest(:,:,:,idx(1:Nval));
    xTestNorm = xTest(:,:,:,idx(Ns-Ntest+1:Ns));
    xTrain = xTrainNorm;
    xVal = xValNorm;
    xTest = xTestNorm;

    globalMin = -0.5;
    scale = 1;
elseif E==4
    E4scenario_choice = 5;
    E4_scenarios = {...
        'Indoor_CloselySpacedUser_2_6GHz'; ...        % : ~3 taps
        'IndoorHall_5GHz'; ...                        % : ~10 taps
        'SemiUrban_CloselySpacedUser_2_6GHz'; ...     % : ~50 taps
        'SemiUrban_300MHz'; ...                       % : ~167 taps
        'SemiUrban_VLA_2_6GHz' ...                    % : ~167 taps
        };

    addpath('COST2100_MATLAB');

    Nt=nTx;
    Nr=1;
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
        2, ...      % Nr
        Ntap_gen, ...     % Ntap
        Ns, ...    % Ntot
        opts);
    
    Hset = Hset(:,:,1,:);
    Horg = delay_to_freq_csi(Hset, Nsub);
    Ht = freq_to_delay_csi(Horg, Ntap);

    Ht_real = real(Ht);
    Ht_imag = imag(Ht);
    
    globalMin = min([Ht_real(:); Ht_imag(:)]);
    globalMax = max([Ht_real(:); Ht_imag(:)]);
    
    scale = globalMax - globalMin;
    
    xTrain = zeros(nTx, maxDelay, numChannels, Ntrain); % [Nt, Ntap, Re/Im, Ns]
    xTrain(:,:,1,:) = real(permute(Ht(Nval+1:Nval+Ntrain, :, :, :), [2, 4, 3, 1]));
    xTrain(:,:,2,:) = imag(permute(Ht(Nval+1:Nval+Ntrain, :, :, :), [2, 4, 3, 1]));

    xVal = zeros(nTx, maxDelay, numChannels, Nval); % [Nt, Ntap, Re/Im, Ns]
    xVal(:,:,1,:) = real(permute(Ht(1:Nval, :, :, :), [2, 4, 3, 1]));
    xVal(:,:,2,:) = imag(permute(Ht(1:Nval, :, :, :), [2, 4, 3, 1]));
    
    xTest = zeros(nTx, maxDelay, numChannels, Ntest);
    xTest(:,:,1,:) = real(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));
    xTest(:,:,2,:) = imag(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));
    
    xTrainNorm = (xTrain - globalMin) / scale;
    xValNorm   = (xVal   - globalMin) / scale;
    xTestNorm  = (xTest  - globalMin) / scale;
elseif E==5
    % Load test data
    load(fullfile("data","DATA_Htestrandom.mat"));
    sampleSize = length(HT);
    xTest = reshape(HT', maxDelay, nTx, numChannels, sampleSize);
    xTest = permute(xTest, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

    idx = randperm(sampleSize, Ns);
    
    xt = permute(xTest(:,:,:,idx), [4, 1, 3, 2])-0.5; % inverse permutation of [2,1,3,4]
    Ht = zeros(Ns, Nt, 1, Ntap);
    Ht(:,:,1,:) = xt(:,:,1,:) + 1j*xt(:,:,2,:);
    
    xTrainNorm = xTest(:,:,:,idx(Nval+1:Nval+Ntrain));
    xValNorm = xTest(:,:,:,idx(1:Nval));
    xTestNorm = xTest(:,:,:,idx(Ns-Ntest+1:Ns));
    xTrain = xTrainNorm;
    xVal = xValNorm;
    xTest = xTestNorm;


    globalMin = -0.5;
    scale = 1;
end




%% Set training parameters and train the network
[CSINet, trainInfo] = trainNetwork(xTrainNorm, xTrainNorm, CSINet, ...
    trainingOptions("adam", ...
    InitialLearnRate=5e-3, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=100, ...
    LearnRateDropFactor=exp(-0.1), ...
    Epsilon=1e-7, ...
    GradientDecayFactor=0.9, ...
    SquaredGradientDecayFactor=0.999, ...
    MaxEpochs=1500, ...
    MiniBatchSize=50, ...
    Shuffle="every-epoch", ...
    Verbose=true, ...
    VerboseFrequency=400, ...
    ValidationData={xValNorm, xValNorm}, ...
    ValidationFrequency=400, ...
    OutputNetwork="best-validation-loss", ...
    Plots="none"));


%%
xHatTrNorm = predict(CSINet, xTrainNorm);

xHatTr = xHatTrNorm * scale + globalMin;

num = squeeze(sum(abs(xTrain - xHatTr).^2, [1 2 3]));   % [Ntest, 1] or [1, Ntest]
den = squeeze(sum(abs(xTrain).^2,       [1 2 3]));    % same shape

nmse = 10*log10(num ./ den);
meanNMSE = mean(nmse);

fprintf("\nAt compression rate 1/%d, NMSE is %f dB (train)\n", 1/compressRate, meanNMSE);


xHatVaNorm = predict(CSINet, xValNorm);

xHatVa = xHatVaNorm * scale + globalMin;

num = squeeze(sum(abs(xVal - xHatVa).^2, [1 2 3]));   % [Ntest, 1] or [1, Ntest]
den = squeeze(sum(abs(xVal).^2,       [1 2 3]));    % same shape

nmse = 10*log10(num ./ den);
meanNMSE = mean(nmse);

fprintf("\nAt compression rate 1/%d, NMSE is %f dB (val)\n", 1/compressRate, meanNMSE);

xHatNorm = predict(CSINet, xTestNorm);

xHat = xHatNorm * scale + globalMin;

num = squeeze(sum(abs(xTest - xHat).^2, [1 2 3]));   % [Ntest, 1] or [1, Ntest]
den = squeeze(sum(abs(xTest).^2,       [1 2 3]));    % same shape

nmse = 10*log10(num ./ den);
meanNMSE = mean(nmse);

fprintf("\nAt compression rate 1/%d, NMSE is %f dB (test)\n", 1/compressRate, meanNMSE);


%% ====== Evaluate R over selected samples ======

xhat_ = permute(xHat, [4, 1, 3, 2]);
Htilde = zeros(Ntest, Nt, 1, Ntap);
Htilde(:,:,1,:) = xhat_(:,:,1,:) + 1j*xhat_(:,:,2,:);

H_org = delay_to_freq_csi(Ht, Nsub);
Htildef = delay_to_freq_csi(Htilde, Nsub);

SNR0_dB = 20;

SNR0 = 10^(SNR0_dB/10);

Nsym = max(min(Nt,Nr)-3,1);
N0   = 1e-2;                  % AWGN variance used in data link evaluation
Nsub = size(H_org,4);

PL = mean(abs(H_org(:)).^2);   % channel gain

Ptot = SNR0*N0*Nt*Nr*Nsub*Nsym/PL;

Neval = Ntest;
Horg_test = H_org(Ns-Ntest+1:Ns, :,:,:);

R_list = zeros(Neval,1);
Rperf_list = zeros(Neval,1);
for ii = 1:Neval

    Horg_t   = Horg_test(ii,:,:,:);    % [Nt,Nr,Nsub]
    Htilde_t = Htildef(ii,:,:,:);  % [Nt,Nr,Nsub]

    Horg_t = reshape(Horg_t, Nt, Nr, Nsub);
    Htilde_t = reshape(Htilde_t, Nt, Nr, Nsub);

    [R_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, Horg_t, Htilde_t, Nsym, Ptot, N0);
    [Rperf_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, Horg_t, Horg_t, Nsym, Ptot, N0);

end

R_avg = mean(R_list);
Rperf_avg = mean(Rperf_list);
fprintf("Average R over %d samples = %.6f bps/Hz\n", Neval, R_avg);
fprintf("Average R over %d samples = %.6f bps/Hz (when perfect CSIR/CSIT)\n", Neval, Rperf_avg);

%% PSVD
fprintf("\n\n===============PSVD===========\n\n");

Htrain = Ht(Nval+1:Nval+Ntrain, :,:,:);
Htest = Ht(Ns-Ntest+1:Ns, :,:,:);

rHtrain = reshape(Htrain, Ntrain, []);
rHtest = reshape(Htest, Ntest, []);

[Vs, sigmaL, ~, ~] = psvd_codebook(rHtrain, compressRate, 0.8, 30);
decoder = pinv(Vs);

Ztrain = rHtrain*Vs;
Ztest = rHtest*Vs;
rtildeHtrain = Ztrain*decoder;
rtildeHtest = Ztest*decoder;

fprintf("Train set: ========= \n");
my_print_error(rHtrain, rtildeHtrain);
fprintf("Test set: ========= \n");
my_print_error(rHtest, rtildeHtest);

tildeHtest = reshape(rtildeHtest, Ntest, Nt, Nr, Ntap);

H_org = delay_to_freq_csi(Ht, Nsub);
Htildef = delay_to_freq_csi(tildeHtest, Nsub);

SNR0_dB = 20;

SNR0 = 10^(SNR0_dB/10);

Nsym = max(min(Nt,Nr)-3,1);
N0   = 1e-2;                  % AWGN variance used in data link evaluation
Nsub = size(H_org,4);

PL = mean(abs(H_org(:)).^2);   % channel gain

Ptot = SNR0*N0*Nt*Nr*Nsub*Nsym/PL;

Neval = Ntest;
idxEval = (Ns-Neval+1):Ns;

opts = struct('use_pinv', true, 'reg_epsilon', 1e-6, 'return_cells', false);

Horg_test = H_org(Ns-Ntest+1:Ns, :,:,:);

R_list = zeros(Neval,1);
Rperf_list = zeros(Neval,1);
for ii = 1:Neval

    Horg_t   = Horg_test(ii,:,:,:);    % [Nt,Nr,Nsub]
    Htilde_t = Htildef(ii,:,:,:);  % [Nt,Nr,Nsub]

    Horg_t = reshape(Horg_t, Nt, Nr, Nsub);
    Htilde_t = reshape(Htilde_t, Nt, Nr, Nsub);

    [R_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, Horg_t, Htilde_t, Nsym, Ptot, N0);
    [Rperf_list(ii), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, Horg_t, Horg_t, Nsym, Ptot, N0);

end

R_avg = mean(R_list);
Rperf_avg = mean(Rperf_list);
fprintf("Average R over %d samples = %.6f bps/Hz\n", Neval, R_avg);
fprintf("Average R over %d samples = %.6f bps/Hz (when perfect CSIR/CSIT)\n", Neval, Rperf_avg);


%% Local functions
function autoencoderLGraph = createCSINet(maxDelay, nTx, numChannels, compressRate)
% Helper function to create CSINet

inputSize = [maxDelay nTx numChannels];
numElements = prod(inputSize);
encodedDim = compressRate*numElements;

autoencoderLGraph = layerGraph([ ...
    % Encoder
    imageInputLayer(inputSize,"Name","Htrunc", ...
    "Normalization","none","Name","input_1")

    convolution2dLayer([3 3],2,"Padding","same","Name","conv2d")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99, ...
    "VarianceDecay",0.99,"Name","batch_normalization")
    leakyReluLayer(0.3,"Name","leaky_re_lu")

    functionLayer(@(x)permute(stripdims(x),[3,2,1,4]), ...
      "Formattable",true,"Acceleratable",true,"Name","Enc_Permute1")

    functionLayer(@(x)dlarray(reshape(x,numChannels*nTx*maxDelay,1,1,[]),'CSSB'), ...
      "Formattable",true,"Acceleratable",true,"Name","Enc_Reshape")

    fullyConnectedLayer(encodedDim,"Name","dense")

    % Decoder
    fullyConnectedLayer(numElements,"Name","dense_1")

    functionLayer(@(x)permute(stripdims(x),[3,2,1,4]),"Formattable",true, ...
    "Acceleratable",true,"Name","Dec_Permute1")

    functionLayer(@(x)dlarray(reshape(x,numChannels,nTx,maxDelay,[]),'CSSB'), ...
    "Formattable",true,"Acceleratable",true,"Name","Dec_Reshape")

    functionLayer(@(x)permute(x,[2,1,3,4]), ...
    "Formattable",true,"Acceleratable",true,"Name","Dec_Permute2")
    ]);

residualLayers1 = [ ...
    convolution2dLayer([3 3],8,"Padding","same","Name","conv2d_1")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_1")
    leakyReluLayer(0.3,"Name","leaky_re_lu_1")

    convolution2dLayer([3 3],16,"Padding","same","Name","conv2d_2")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_2")
    leakyReluLayer(0.3,"Name","leaky_re_lu_2")

    convolution2dLayer([3 3],2,"Padding","same","Name","Res_Conv_1_3")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_3")

    additionLayer(2,"Name","add")

    leakyReluLayer(0.3,"Name","leaky_re_lu_3")
    ];

autoencoderLGraph = addLayers(autoencoderLGraph,residualLayers1);
autoencoderLGraph = connectLayers(autoencoderLGraph,"Dec_Permute2","conv2d_1");
autoencoderLGraph = connectLayers(autoencoderLGraph,"Dec_Permute2","add/in2");

residualLayers2 = [ ...
    convolution2dLayer([3 3],8,"Padding","same","Name","conv2d_4")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_4")
    leakyReluLayer(0.3,"Name","leaky_re_lu_4")

    convolution2dLayer([3 3],16,"Padding","same","Name","conv2d_5")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_5")
    leakyReluLayer(0.3,"Name","leaky_re_lu_5")

    convolution2dLayer([3 3],2,"Padding","same","Name","conv2d_6")
    batchNormalizationLayer("Epsilon",0.001,"MeanDecay",0.99,"VarianceDecay",0.99,"Name","batch_normalization_6")

    additionLayer(2,"Name","add_1")

    leakyReluLayer(0.3,"Name","leaky_re_lu_6")
    ];

autoencoderLGraph = addLayers(autoencoderLGraph,residualLayers2);
autoencoderLGraph = connectLayers(autoencoderLGraph,"leaky_re_lu_3","conv2d_4");
autoencoderLGraph = connectLayers(autoencoderLGraph,"leaky_re_lu_3","add_1/in2");


autoencoderLGraph = addLayers(autoencoderLGraph, ...
    [convolution2dLayer([3 3],2,"Padding","same","Name","conv2d_7") ...
    sigmoidLayer("Name","conv2d_7_sigmoid") ...
    regressionLayer("Name","RegressionLayer_conv2d_7")]);

autoencoderLGraph = ...
    connectLayers(autoencoderLGraph,"leaky_re_lu_6","conv2d_7");
end