function loss = mmdRbfLoss(X, Y, sigma)
% X, Y: [L, B], format "CB"
% Computes biased MMD^2 with RBF kernel.

    X = stripdims(X);
    Y = stripdims(Y);

    Kxx = rbfKernelMatrix(X, X, sigma);
    Kyy = rbfKernelMatrix(Y, Y, sigma);
    Kxy = rbfKernelMatrix(X, Y, sigma);

    loss = mean(Kxx, 'all') + mean(Kyy, 'all') - 2 * mean(Kxy, 'all');
end

function K = rbfKernelMatrix(X, Y, sigma)
% X: [d, n], Y: [d, m]
    XX = sum(X.^2, 1);
    YY = sum(Y.^2, 1);

    XXcol = reshape(XX, [numel(XX),1]);
    YYrow = reshape(YY, [1,numel(YY)]);

    % pairwise squared distances
    % D2 = reshape(XX, [], 1) + reshape(YY, 1, []) - 2 * (X' * Y);
    D2 = XXcol + YYrow - 2 * (X' * Y);
    D2 = max(D2, 0);

    K = exp(-D2 / (2 * sigma^2));
end