function [gradEnc, gradDec, stateEnc, stateDec, totalLoss, reconLoss, domainLoss] = ...
    modelGradientsDDAImCsiNetS(encoderNet, decoderNet, dlXField, dlZSource, config, ...
                               useBinarization, alphaDomainLoss, kernelSigma)

    % Field encoder forward
    [dlZField, stateEnc] = forward(encoderNet, dlXField);

    if useBinarization
        dlZFieldQ = steStochasticBinarize(dlZField);
    else
        dlZFieldQ = dlZField;
    end

    % Field reconstruction
    [dlYField, stateDec] = forward(decoderNet, dlZFieldQ);

    % Reconstruction loss on field samples
    reconLoss = negativeCosineSimilarityLossMultiSubband( ...
        dlYField, dlXField, config.nTx, config.nSubbands);

    % Domain loss on codewords: MMD(source latent, field latent)
    domainLoss = mmdRbfLoss(dlZSource, dlZFieldQ, kernelSigma);

    totalLoss = reconLoss + alphaDomainLoss * domainLoss;

    gradEnc = dlgradient(totalLoss, encoderNet.Learnables);
    gradDec = dlgradient(totalLoss, decoderNet.Learnables);
end