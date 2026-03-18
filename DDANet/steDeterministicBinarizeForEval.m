function y = steDeterministicBinarizeForEval(x)
% For evaluation, deterministic sign binarization
    xData = extractdata(x);
    q = sign(xData);
    q(q == 0) = 1;
    y = dlarray(q, dims(x));
end