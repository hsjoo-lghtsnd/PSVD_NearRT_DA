function loss = negativeCosineSimilarityLoss(dlY, dlX, nTx)
% dlY, dlX: [2*nTx, batch], format "CB"

    yReal = dlY(1:nTx, :);
    yImag = dlY(nTx+1:end, :);
    xReal = dlX(1:nTx, :);
    xImag = dlX(nTx+1:end, :);

    yC = complex(yReal, yImag);
    xC = complex(xReal, xImag);

    num = abs(sum(conj(yC) .* xC, 1));
    den = sqrt(sum(abs(yC).^2, 1)) .* sqrt(sum(abs(xC).^2, 1)) + 1e-12;

    rho = num ./ den;    % cosine similarity per sample
    loss = -mean(rho);
end