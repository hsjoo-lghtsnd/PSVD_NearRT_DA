function avgCos = avg_cosine_similarity_matrix(Yhat, Xtrue, Nt, Nsband)
% Yhat, Xtrue : [2*Nt*Nsband, N]
% avgCos      : scalar average cosine similarity over samples
%
% Interprets each sample as stacked complex subband eigenvectors.

    [D, N] = size(Yhat);
    assert(all(size(Xtrue) == [D, N]), 'Yhat and Xtrue must have same size.');
    assert(D == 2*Nt*Nsband, 'Dimension mismatch.');

    rho = zeros(1, N);

    for n = 1:N
        Vhat = unpack_implicit_features(Yhat(:,n), Nt, Nsband, true);   % [Nt, Nsband, 1]
        Vtru = unpack_implicit_features(Xtrue(:,n), Nt, Nsband, true);

        csum = 0;
        for s = 1:Nsband
            vh = Vhat(:,s,1);
            vt = Vtru(:,s,1);

            num = abs(vh' * vt);
            den = norm(vh) * norm(vt) + 1e-12;
            csum = csum + num / den;
        end
        rho(n) = csum / Nsband;
    end

    avgCos = mean(rho);
end