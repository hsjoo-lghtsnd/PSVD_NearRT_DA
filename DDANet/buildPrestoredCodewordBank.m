function ZSourceBank = buildPrestoredCodewordBank(encoderNet, XSourceBank, useGPU, useBinarization)
% Build prestored/source codeword bank for DDA domain loss.
%
% XSourceBank: [2*nTx, Nbank]
% ZSourceBank: [L, Nbank]

    dlX = dlarray(single(XSourceBank), "CB");
    if useGPU
        dlX = gpuArray(dlX);
    end

    dlZ = forward(encoderNet, dlX);

    if useBinarization
        dlZ = steDeterministicBinarizeForEval(dlZ);
    end

    ZSourceBank = gather(extractdata(dlZ));
end