function y = steStochasticBinarize(x)
% STE stochastic binarization for x in [-1,1].
% Forward:
%   y \in {-1, +1}, with P(y=+1)=(1+x)/2
% Backward:
%   dy/dx = 1 (straight-through style)

    xData = extractdata(x);   % detached underlying numeric/gpuArray
    p = (1 + xData) / 2;
    u = rand(size(xData), 'like', xData);
    q = ones(size(xData), 'like', xData);
    q(u > p) = -1;

    % STE trick:
    % forward value becomes q, gradient wrt x stays 1
    y = x + (dlarray(q, dims(x)) - dlarray(xData, dims(x)));
end