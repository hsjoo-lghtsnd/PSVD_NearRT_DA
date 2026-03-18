function [encoderNet, decoderNet, history] = finetuneDDAImCsiNetS( ...
    encoderNet, decoderNet, ZSourceBank, XFieldTrain, XFieldVal, nTx, opts)
% DDA-Net style unsupervised finetuning for ImCsiNet-s custom loop.
%
% encoderNet, decoderNet : pretrained networks
% ZSourceBank            : [L, Nbank] prestored/source latent codeword bank
% XFieldTrain, XFieldVal : [2*nTx, N]
%
% opts:
%   .MaxEpochs
%   .MiniBatchSize
%   .InitialLearnRate
%   .UseGPU
%   .UseBinarization
%   .AlphaDomainLoss
%   .KernelSigma
%   .GradDecay
%   .SqGradDecay
%   .Verbose
%   .ValFrequency

    arguments
        encoderNet dlnetwork
        decoderNet dlnetwork
        ZSourceBank single
        XFieldTrain double
        XFieldVal double
        nTx (1,1) double {mustBePositive, mustBeInteger}
        opts struct = struct()
    end
    
    if ~isfield(opts, 'MaxEpochs') || isempty(opts.MaxEpochs)
        opts.MaxEpochs = 100;
    end
    if ~isfield(opts, 'MiniBatchSize') || isempty(opts.MiniBatchSize)
        opts.MiniBatchSize = 100;
    end
    if ~isfield(opts, 'InitialLearnRate') || isempty(opts.InitialLearnRate)
        opts.InitialLearnRate = 1e-2;
    end
    if ~isfield(opts, 'UseGPU') || isempty(opts.UseGPU)
        opts.UseGPU = false;
    end
    if ~isfield(opts, 'UseBinarization') || isempty(opts.UseBinarization)
        opts.UseBinarization = true;
    end
    if ~isfield(opts, 'AlphaDomainLoss') || isempty(opts.AlphaDomainLoss)
        opts.AlphaDomainLoss = 50;
    end
    if ~isfield(opts, 'KernelSigma') || isempty(opts.KernelSigma)
        opts.KernelSigma = 1.0;
    end
    if ~isfield(opts, 'GradDecay') || isempty(opts.GradDecay)
        opts.GradDecay = 0.9;
    end
    if ~isfield(opts, 'SqGradDecay') || isempty(opts.SqGradDecay)
        opts.SqGradDecay = 0.999;
    end
    if ~isfield(opts, 'Verbose') || isempty(opts.Verbose)
        opts.Verbose = true;
    end
    if ~isfield(opts, 'ValFrequency') || isempty(opts.ValFrequency)
        opts.ValFrequency = 10;
    end

    if opts.UseGPU
        encoderNet = dlupdate(@gpuArray, encoderNet);
        decoderNet = dlupdate(@gpuArray, decoderNet);
        ZSourceBank = gpuArray(single(ZSourceBank));
    else
        ZSourceBank = single(ZSourceBank);
    end

    % Field minibatch queue
    mbq = minibatchqueue( ...
        arrayDatastore(XFieldTrain, IterationDimension=2), ...
        MiniBatchSize=opts.MiniBatchSize, ...
        MiniBatchFcn=@(x) preprocessMiniBatch(x, opts.UseGPU), ...
        MiniBatchFormat="CB", ...
        PartialMiniBatch="discard");

    trailingAvgEnc = [];
    trailingAvgSqEnc = [];
    trailingAvgDec = [];
    trailingAvgSqDec = [];

    iteration = 0;

    history.iteration = [];
    history.totalLoss = [];
    history.reconLoss = [];
    history.domainLoss = [];
    history.valFieldLoss = [];
    history.valFieldCos = [];

    config = struct();
    config.nTx = nTx;

    for epoch = 1:opts.MaxEpochs
        shuffle(mbq);

        while hasdata(mbq)
            iteration = iteration + 1;
            dlXField = next(mbq);   % [2*nTx, B]

            % sample same-size source latent minibatch from bank
            batchSize = size(dlXField, 2);
            idx = randperm(size(ZSourceBank, 2), batchSize);
            dlZSource = dlarray(ZSourceBank(:, idx), "CB");

            [gradEnc, gradDec, stateEnc, stateDec, totalLoss, reconLoss, domainLoss] = dlfeval( ...
                @modelGradientsDDAImCsiNetS, encoderNet, decoderNet, dlXField, dlZSource, config, ...
                opts.UseBinarization, opts.AlphaDomainLoss, opts.KernelSigma);

            encoderNet.State = stateEnc;
            decoderNet.State = stateDec;

            [encoderNet, trailingAvgEnc, trailingAvgSqEnc] = adamupdate( ...
                encoderNet, gradEnc, trailingAvgEnc, trailingAvgSqEnc, ...
                iteration, opts.InitialLearnRate, opts.GradDecay, opts.SqGradDecay);

            [decoderNet, trailingAvgDec, trailingAvgSqDec] = adamupdate( ...
                decoderNet, gradDec, trailingAvgDec, trailingAvgSqDec, ...
                iteration, opts.InitialLearnRate, opts.GradDecay, opts.SqGradDecay);

            history.iteration(end+1,1) = iteration;
            history.totalLoss(end+1,1) = double(gather(extractdata(totalLoss)));
            history.reconLoss(end+1,1) = double(gather(extractdata(reconLoss)));
            history.domainLoss(end+1,1) = double(gather(extractdata(domainLoss)));

            if mod(iteration, opts.ValFrequency) == 0 && ~isempty(XFieldVal)
                [valLoss, valCos] = evaluateImCsiNetS( ...
                    encoderNet, decoderNet, XFieldVal, struct('nTx', nTx), ...
                    opts.UseGPU, opts.UseBinarization);

                history.valFieldLoss(end+1,1) = valLoss;
                history.valFieldCos(end+1,1) = valCos;

                if opts.Verbose
                    fprintf(['DDA Epoch %d, Iter %d, Total %.6f, Recon %.6f, Domain %.6f, ' ...
                             'ValFieldLoss %.6f, ValFieldCos %.6f\n'], ...
                             epoch, iteration, ...
                             history.totalLoss(end), history.reconLoss(end), history.domainLoss(end), ...
                             valLoss, valCos);
                end
            end
        end
    end
end