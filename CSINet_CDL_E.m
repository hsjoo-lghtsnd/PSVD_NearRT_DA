% This MATLAB script trains CSINet model proposed in
% 'Chao-Kai Wen, Wan-Ting Shih, and Shi Jin, "Deep learning for massive MIMO CSI feedback,”
% IEEE Wireless Communications Letters, 2018. [Online]. Available: https://ieeexplore.ieee.org/document/8322184/.'
% using in MATLAB®.

% Set network parameters
maxDelay = 32;
nTx = 32;
numChannels = 2;
compressRate = 1/64; % 1/4 | 1/16 | 1/32 | 1/64
environment = "indoor"; % "indoor" | "outdoor"

% Create CSINet deep network
CSINet = createCSINet(maxDelay, nTx, numChannels, compressRate);

% Analyze CSINet architecture visually
analyzeNetwork(CSINet);

%% Data loading
% % Load training data
% load(fullfile("data","DATA_Htrain"+extractBefore(environment,"door")+".mat"));
% sampleSize = length(HT);
% xTrain = reshape(HT',maxDelay, nTx, numChannels, sampleSize);
% xTrain = permute(xTrain, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize
% 
% % Load validation data
% load(fullfile("data","DATA_Hval"+extractBefore(environment,"door")+".mat"));
% sampleSize = length(HT);
% xVal = reshape(HT', maxDelay, nTx, numChannels, sampleSize);
% xVal = permute(xVal, [2, 1, 3, 4]); % permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

Ns = 2000;
SCS = 15;                      % kHz
Nsub = 624;                    % 52 RB * 12 subcarriers
TxArraySize = [4 4 2 1 1];     % [rows, cols, pol, panelRows, panelCols]
RxArraySize = [1 1 1 1 1];
delaySpread = 300e-9;          % seconds
cdlProfile = "CDL-E";
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

Ht = freq_to_delay_csi(Hobs, Ntap); % [Ns, Nt, Nr, Ntap]
disp(size(Ht));

%%
Ntrain = 100;
Nval = 100;
Ntest = 1000;

Ht_real = real(Ht);
Ht_imag = imag(Ht);

globalMin = min([Ht_real(:); Ht_imag(:)]);
globalMax = max([Ht_real(:); Ht_imag(:)]);

scale = globalMax - globalMin;

xTrain = zeros(nTx, maxDelay, numChannels, Ntrain); % [Nt, Ntap, Re/Im, Ns]
xTrain(:,:,1,:) = real(permute(Ht(1:Ntrain, :, :, :), [2, 4, 3, 1]));
xTrain(:,:,2,:) = imag(permute(Ht(1:Ntrain, :, :, :), [2, 4, 3, 1]));

xVal = zeros(nTx, maxDelay, numChannels, Ntrain);
xVal(:,:,1,:) = real(permute(Ht(Ntrain+1:Ntrain+Nval, :, :, :), [2, 4, 3, 1]));
xVal(:,:,2,:) = imag(permute(Ht(Ntrain+1:Ntrain+Nval, :, :, :), [2, 4, 3, 1]));

xTest = zeros(nTx, maxDelay, numChannels, Ntest);
xTest(:,:,1,:) = real(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));
xTest(:,:,2,:) = imag(permute(Ht(Ns-Ntest+1:Ns, :, :, :), [2, 4, 3, 1]));

% permute xTrain to nTx-by-maxDelay-by-numChannels-by-batchSize

xTrainNorm = (xTrain - globalMin) / scale;
xValNorm   = (xVal   - globalMin) / scale;
xTestNorm  = (xTest  - globalMin) / scale;

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
    MiniBatchSize=500, ...
    Shuffle="every-epoch", ...
    Verbose=true, ...
    VerboseFrequency=400, ...
    ValidationData={xValNorm, xValNorm}, ...
    ValidationFrequency=400, ...
    OutputNetwork="best-validation-loss", ...
    Plots="none"));


%%
xHatNorm = predict(CSINet, xTestNorm);

xHat = xHatNorm * scale + globalMin;

num = squeeze(sum(abs(xTest - xHat).^2, [1 2 3]));   % [Ntest, 1] or [1, Ntest]
den = squeeze(sum(abs(xTest).^2,       [1 2 3]));    % same shape

nmse = 10*log10(num ./ den);
meanNMSE = mean(nmse);

fprintf("\nAt compression rate 1/%d, NMSE is %f dB\n", 1/compressRate, meanNMSE);

%% Train NMSE
xHatTrainNorm = predict(CSINet, xTrainNorm);

xHatTrain = xHatTrainNorm * scale + globalMin;

numTrain = squeeze(sum(abs(xTrain - xHatTrain).^2, [1 2 3]));
denTrain = squeeze(sum(abs(xTrain).^2,           [1 2 3]));

nmseTrain = 10*log10(numTrain ./ denTrain);
meanNMSEtrain = mean(nmseTrain);

fprintf("\nTrain NMSE at compression rate 1/%d: %f dB\n", ...
    1/compressRate, meanNMSEtrain);

%% Validation NMSE
xHatValNorm = predict(CSINet, xValNorm);

xHatVal = xHatValNorm * scale + globalMin;

numVal = squeeze(sum(abs(xVal - xHatVal).^2, [1 2 3]));
denVal = squeeze(sum(abs(xVal).^2,          [1 2 3]));

nmseVal = 10*log10(numVal ./ denVal);
meanNMSEval = mean(nmseVal);

fprintf("\nValidation NMSE at compression rate 1/%d: %f dB\n", ...
    1/compressRate, meanNMSEval);

%% Extract encoder output (codeword) directly from trained network

Ztrain = activations(CSINet, xTrain, "dense", "OutputAs", "rows");
Zval   = activations(CSINet, xVal,   "dense", "OutputAs", "rows");

disp(size(Ztrain));   % expected: [Ntrain, encodedDim]
disp(size(Zval));     % expected: [Nval, encodedDim]

%% Save trained network
savedNetFileName = "model_CsiNet_"+environment+"_dim"+num2str(maxDelay*nTx*numChannels*compressRate)+".mat";
save(savedNetFileName, "CSINet")

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