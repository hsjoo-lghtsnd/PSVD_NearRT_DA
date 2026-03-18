function Vstack = unpack_implicit_features(X, Nt, K, renormalize)
% unpack_implicit_features
% Input:
%   X          : [2*Nt*K, Ns] real
% Output:
%   Vstack     : [Nt, K, Ns] complex
%
% renormalize:
%   If true, each vector is normalized to unit norm.

    arguments
        X
        Nt (1,1) double {mustBePositive, mustBeInteger}
        K (1,1) double {mustBePositive, mustBeInteger}
        renormalize (1,1) logical = true
    end

    [D, Ns] = size(X);
    assert(D == 2*Nt*K, 'Input dimension mismatch.');

    Vstack = complex(zeros(Nt, K, Ns, 'single'));

    for n = 1:Ns
        for k = 1:K
            idx = (k-1)*2*Nt + (1:2*Nt);
            block = X(idx, n);

            v = complex(block(1:Nt), block(Nt+1:end));

            if renormalize
                nv = norm(v);
                if nv > 0
                    v = v / nv;
                end
            end

            Vstack(:,k,n) = v;
        end
    end
end