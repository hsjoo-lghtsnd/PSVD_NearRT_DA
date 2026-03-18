function rho = cosineSimilarityPerSampleMultiSubband(dlY, dlX, nTx, nSubbands)
    batchSize = size(dlX, 2);
    rho = zeros(1, batchSize, 'like', dlX);

    for b = 1:batchSize
        yb = dlY(:,b);
        xb = dlX(:,b);

        csum = 0;
        for s = 1:nSubbands
            idx = (s-1)*2*nTx + (1:2*nTx);

            yblock = yb(idx);
            xblock = xb(idx);

            vy = complex(yblock(1:nTx), yblock(nTx+1:end));
            vx = complex(xblock(1:nTx), xblock(nTx+1:end));

            num = abs(sum(conj(vy).*vx, 1));
            den = sqrt(sum(abs(vy).^2,1)) * sqrt(sum(abs(vx).^2,1)) + 1e-12;
            csum = csum + num / den;
        end
        rho(b) = csum / nSubbands;
    end
end