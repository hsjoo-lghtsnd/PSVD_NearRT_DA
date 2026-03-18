function [valLoss, valCos] = evaluateImCsiNetM(encoderNet, decoderNet, XVal, config, useGPU, useBinarization)

    dlX = dlarray(single(XVal), "CB");
    if useGPU
        dlX = gpuArray(dlX);
    end

    dlZ = forward(encoderNet, dlX);

    if useBinarization
        dlZq = steDeterministicBinarizeForEval(dlZ);
    else
        dlZq = dlZ;
    end

    dlY = forward(decoderNet, dlZq);

    loss = negativeCosineSimilarityLossMultiSubband( ...
        dlY, dlX, config.nTx, config.nSubbands);
    valLoss = double(gather(extractdata(loss)));

    rho = cosineSimilarityPerSampleMultiSubband( ...
        dlY, dlX, config.nTx, config.nSubbands);
    valCos = double(gather(mean(extractdata(rho))));
end