function [encoderNet, decoderNet, config] = createImCsiNetM832( ...
    nTx, nSubbands, compressRate, varargin)
% createImCsiNetM832
% Multi-subband FC autoencoder for DDA-Net / ImCsiNet-m style implicit CSI.
%
% Input representation:
%   Vstack \in C^{Nt x Ns}
%   -> stack eigenvectors vertically
%   -> concatenate real and imaginary parts
%   -> x \in R^{2*Nt*Ns}
%
% For DDA-Net faithful setting:
%   Nt = 32, Ns = 13 => inputDim = 832
%   CR = 1/64 => encodedDim = round(832/64) = 13
%
% Architecture:
%   Encoder:
%       FC(hidden1) -> BN -> LeakyReLU
%       FC(hidden2) -> BN -> LeakyReLU
%       FC(L)       -> BN -> Tanh
%   Decoder:
%       FC(hidden2) -> BN -> LeakyReLU
%       FC(hidden1) -> BN -> LeakyReLU
%       FC(inputDim)-> BN -> Tanh
%
% This matches the ImCsiNet-m design principle:
%   - FC-based encoder/decoder
%   - input dimension 2*Nt*Ns
%   - latent dimension L = alpha * 2*Nt*Ns
%   - decoder output reshaped back to Vhat_stack
%
% Optional name-value pairs:
%   'HiddenDim1'      : first hidden dimension
%   'HiddenDim2'      : second hidden dimension
%   'Leak'            : leaky ReLU slope
%   'UseQuantLayer'   : add a forward-only uniform quantization placeholder
%   'QuantBits'       : quantization bits B (used only if UseQuantLayer=true)
%
% Outputs:
%   encoderNet : dlnetwork
%   decoderNet : dlnetwork
%   config     : struct with dimensions

    p = inputParser;
    addParameter(p, 'HiddenDim1', [], @(x)isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'HiddenDim2', [], @(x)isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'Leak', 0.3, @(x)isnumeric(x) && isscalar(x) && x >= 0);
    addParameter(p, 'UseQuantLayer', false, @(x)islogical(x) || isnumeric(x));
    addParameter(p, 'QuantBits', 4, @(x)isnumeric(x) && isscalar(x) && x >= 1);
    parse(p, varargin{:});

    leak = p.Results.Leak;
    useQuantLayer = logical(p.Results.UseQuantLayer);
    quantBits = round(p.Results.QuantBits);

    inputDim   = 2 * nTx * nSubbands;
    encodedDim = max(1, round(compressRate * inputDim));

    % Default hidden widths:
    % keep the "same-depth, wider-than-single-RB" spirit of ImCsiNet-m.
    % 832-d input often works well with ~2x and ~1x widths.
    if isempty(p.Results.HiddenDim1)
        hiddenDim1 = 2 * inputDim;       % e.g., 1664 for 832-d input
    else
        hiddenDim1 = round(p.Results.HiddenDim1);
    end

    if isempty(p.Results.HiddenDim2)
        hiddenDim2 = inputDim;           % e.g., 832 for 832-d input
    else
        hiddenDim2 = round(p.Results.HiddenDim2);
    end

    % -------------------------
    % Encoder
    % -------------------------
    encLayers = [
        featureInputLayer(inputDim, ...
            "Normalization","none", ...
            "Name","enc_input")

        fullyConnectedLayer(hiddenDim1, "Name","enc_fc1")
        batchNormalizationLayer("Name","enc_bn1")
        leakyReluLayer(leak, "Name","enc_lrelu1")

        fullyConnectedLayer(hiddenDim2, "Name","enc_fc2")
        batchNormalizationLayer("Name","enc_bn2")
        leakyReluLayer(leak, "Name","enc_lrelu2")

        fullyConnectedLayer(encodedDim, "Name","enc_fc3")
        batchNormalizationLayer("Name","enc_bn3")
        tanhLayer("Name","enc_tanh")
    ];

    encLG = layerGraph(encLayers);

    if useQuantLayer
        quantLayer = functionLayer( ...
            @(x) uniformQuantizeForward(x, quantBits), ...
            "Name","enc_uniform_quant", ...
            "Formattable",true, ...
            "Acceleratable",true);
        encLG = addLayers(encLG, quantLayer);
        encLG = connectLayers(encLG, "enc_tanh", "enc_uniform_quant");
    end

    encoderNet = dlnetwork(encLG);

    % -------------------------
    % Decoder
    % -------------------------
    decInputName = "dec_input";

    if useQuantLayer
        decoderInputDim = encodedDim;  % same latent size after quantization
    else
        decoderInputDim = encodedDim;
    end

    decLayers = [
        featureInputLayer(decoderInputDim, ...
            "Normalization","none", ...
            "Name",decInputName)

        fullyConnectedLayer(hiddenDim2, "Name","dec_fc1")
        batchNormalizationLayer("Name","dec_bn1")
        leakyReluLayer(leak, "Name","dec_lrelu1")

        fullyConnectedLayer(hiddenDim1, "Name","dec_fc2")
        batchNormalizationLayer("Name","dec_bn2")
        leakyReluLayer(leak, "Name","dec_lrelu2")

        fullyConnectedLayer(inputDim, "Name","dec_fc3")
        batchNormalizationLayer("Name","dec_bn3")
        tanhLayer("Name","dec_tanh")
    ];

    decoderNet = dlnetwork(layerGraph(decLayers));

    % -------------------------
    % Config
    % -------------------------
    config = struct();
    config.nTx = nTx;
    config.nSubbands = nSubbands;
    config.inputDim = inputDim;
    config.encodedDim = encodedDim;
    config.hiddenDim1 = hiddenDim1;
    config.hiddenDim2 = hiddenDim2;
    config.leak = leak;
    config.useQuantLayer = useQuantLayer;
    config.quantBits = quantBits;
end


function y = uniformQuantizeForward(x, B)
% Forward-only uniform quantization placeholder in [-1,1].
% For strict training reproduction, use a custom STE quantizer in the
% custom training loop instead of relying on this functionLayer.
%
% Qu(x) = round(x * 2^(B-1)) / 2^(B-1)

    scale = 2^(B-1);
    xData = extractdata(x);
    xData = max(min(xData, 1), -1);
    qData = round(xData * scale) / scale;
    y = dlarray(qData, dims(x));
end