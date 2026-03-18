function [gradEnc, gradDec, stateEnc, stateDec, loss] = modelGradientsImCsiNetM( ...
    encoderNet, decoderNet, dlX, config, useBinarization)

    [dlZ, stateEnc] = forward(encoderNet, dlX);

    if useBinarization
        dlZq = steStochasticBinarize(dlZ);
    else
        dlZq = dlZ;
    end

    [dlY, stateDec] = forward(decoderNet, dlZq);

    loss = negativeCosineSimilarityLossMultiSubband( ...
        dlY, dlX, config.nTx, config.nSubbands);

    gradEnc = dlgradient(loss, encoderNet.Learnables);
    gradDec = dlgradient(loss, decoderNet.Learnables);
end