function Yhat = predictImCsiNetAutoencoder(encoderNet, decoderNet, X, useGPU, useBinarization)
% X    : [D, N]
% Yhat : [D, N]

    dlX = dlarray(single(X), "CB");
    if useGPU
        dlX = gpuArray(dlX);
    end

    dlZ = forward(encoderNet, dlX);

    if useBinarization
        dlZ = steDeterministicBinarizeForEval(dlZ);
    end

    dlY = forward(decoderNet, dlZ);
    Yhat = gather(extractdata(dlY));
end