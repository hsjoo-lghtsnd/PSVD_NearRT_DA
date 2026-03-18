% function [encoderNet, decoderNet, history] = pretrainExistingImCsiNetForDDA(...
%     encoderNet, decoderNet, XSourceTrain, XSourceVal, nTx, nSubbands, t_opts)
function [encoderNet, decoderNet, history] = pretrainExistingImCsiNetForDDA(encoderNet, decoderNet, XSourceTrain, XSourceVal, nTx, nSubbands, t_opts)
% Pretrain an already-created ImCsiNet-m style autoencoder.
%
% encoderNet, decoderNet : dlnetwork objects already created
% XSourceTrain, XSourceVal : [2*nTx*nSubbands, N]
%
% Uses only reconstruction loss (negative cosine similarity).

    arguments
        encoderNet dlnetwork
        decoderNet dlnetwork
        XSourceTrain double
        XSourceVal double
        nTx (1,1) double {mustBePositive, mustBeInteger}
        nSubbands (1,1) double {mustBePositive, mustBeInteger}
        t_opts struct = struct()
    end

    if ~isfield(t_opts, 'MaxEpochs') || isempty(t_opts.MaxEpochs)
        t_opts.MaxEpochs = 200;
    end
    if ~isfield(t_opts, 'MiniBatchSize') || isempty(t_opts.MiniBatchSize)
        t_opts.MiniBatchSize = 1024;
    end
    if ~isfield(t_opts, 'InitialLearnRate') || isempty(t_opts.InitialLearnRate)
        t_opts.InitialLearnRate = 1e-3;
    end
    if ~isfield(t_opts, 'UseGPU') || isempty(t_opts.UseGPU)
        t_opts.UseGPU = false;
    end
    if ~isfield(t_opts, 'Verbose') || isempty(t_opts.Verbose)
        t_opts.Verbose = true;
    end
    if ~isfield(t_opts, 'DoValidation') || isempty(t_opts.DoValidation)
        t_opts.DoValidation = true;
    end
    if ~isfield(t_opts, 'ValFrequency') || isempty(t_opts.ValFrequency)
        t_opts.ValFrequency = 20;
    end
    if ~isfield(t_opts, 'UseBinarization') || isempty(t_opts.UseBinarization)
        t_opts.UseBinarization = false;
    end
    if ~isfield(t_opts, 'GradDecay') || isempty(t_opts.GradDecay)
        t_opts.GradDecay = 0.9;
    end
    if ~isfield(t_opts, 'SqGradDecay') || isempty(t_opts.SqGradDecay)
        t_opts.SqGradDecay = 0.999;
    end

    inputDimExpected = 2 * nTx * nSubbands;
    assert(size(XSourceTrain,1) == inputDimExpected, ...
        'XSourceTrain first dimension must be 2*nTx*nSubbands.');
    assert(isempty(XSourceVal) || size(XSourceVal,1) == inputDimExpected, ...
        'XSourceVal first dimension must be 2*nTx*nSubbands.');

    if t_opts.UseGPU
        encoderNet = dlupdate(@gpuArray, encoderNet);
        decoderNet = dlupdate(@gpuArray, decoderNet);
    end

    mbq = minibatchqueue( ...
        arrayDatastore(XSourceTrain, IterationDimension=2), ...
        MiniBatchSize=t_opts.MiniBatchSize, ...
        MiniBatchFcn=@(x) preprocessMiniBatch(x, t_opts.UseGPU), ...
        MiniBatchFormat="CB", ...
        PartialMiniBatch="discard");

    trailingAvgEnc = [];
    trailingAvgSqEnc = [];
    trailingAvgDec = [];
    trailingAvgSqDec = [];

    iteration = 0;

    history.iteration = [];
    history.trainLoss = [];
    history.valLoss = [];
    history.valCosine = [];

    config = struct();
    config.nTx = nTx;
    config.nSubbands = nSubbands;

    for epoch = 1:t_opts.MaxEpochs
        shuffle(mbq);

        while hasdata(mbq)
            iteration = iteration + 1;
            dlX = next(mbq);

            [gradEnc, gradDec, stateEnc, stateDec, lossVal] = dlfeval( ...
                @modelGradientsImCsiNetM, encoderNet, decoderNet, dlX, config, t_opts.UseBinarization);

            encoderNet.State = stateEnc;
            decoderNet.State = stateDec;

            [encoderNet, trailingAvgEnc, trailingAvgSqEnc] = adamupdate( ...
                encoderNet, gradEnc, trailingAvgEnc, trailingAvgSqEnc, ...
                iteration, t_opts.InitialLearnRate, t_opts.GradDecay, t_opts.SqGradDecay);

            [decoderNet, trailingAvgDec, trailingAvgSqDec] = adamupdate( ...
                decoderNet, gradDec, trailingAvgDec, trailingAvgSqDec, ...
                iteration, t_opts.InitialLearnRate, t_opts.GradDecay, t_opts.SqGradDecay);

            history.iteration(end+1,1) = iteration;
            history.trainLoss(end+1,1) = double(gather(extractdata(lossVal)));

            doValNow = t_opts.DoValidation && ~isempty(XSourceVal) && mod(iteration, t_opts.ValFrequency) == 0;
            if doValNow
                [valLoss, valCos] = evaluateImCsiNetM( ...
                    encoderNet, decoderNet, XSourceVal, config, t_opts.UseGPU, t_opts.UseBinarization);

                history.valLoss(end+1,1) = valLoss;
                history.valCosine(end+1,1) = valCos;

                if t_opts.Verbose
                    fprintf('Pretrain Epoch %d, Iter %d, TrainLoss %.6f, ValLoss %.6f, ValCos %.6f\n', ...
                        epoch, iteration, history.trainLoss(end), valLoss, valCos);
                end
            end
        end
    end
end