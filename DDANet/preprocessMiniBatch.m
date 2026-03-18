function dlX = preprocessMiniBatch(xCell, useGPU)
% xCell comes from arrayDatastore with IterationDimension=2
% It is usually a cell array of mini-batch columns.

    if iscell(xCell)
        X = cat(2, xCell{:});
    else
        X = xCell;
    end

    dlX = dlarray(single(X), "CB");

    if useGPU
        dlX = gpuArray(dlX);
    end
end