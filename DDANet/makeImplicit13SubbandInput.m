function X = makeImplicit13SubbandInput(Hf)
% Hf: [Ns, Nt, Nr, Nsub] complex
% X : [13*2*Nt, Ns] real
%
% For each sample:
%   - divide Nsub into 13 contiguous subbands
%   - average H over each subband
%   - compute dominant right singular vector v_k
%   - stack [real(v_k); imag(v_k)] for k=1,...,13

    [Ns, Nt, Nr, Nsub] = size(Hf);
    assert(mod(Nsub,13)==0, 'Nsub must be divisible by 13.');

    subbandSize = Nsub / 13;
    X = zeros(13 * 2 * Nt, Ns, 'single');

    for n = 1:Ns
        feat = zeros(13 * 2 * Nt, 1, 'single');

        for k = 1:13
            idx1 = (k-1)*subbandSize + 1;
            idx2 = k*subbandSize;

            % Hblk: [Nt, Nr, subbandSize]
            Hblk = squeeze(Hf(n,:,:,idx1:idx2));

            % representative channel for subband
            % convert to [Nr, Nt] by averaging over subcarriers
            Hbar = mean(permute(Hblk, [2 1 3]), 3);   % [Nr, Nt]

            [~,~,V] = svd(Hbar, 'econ');
            v = V(:,1);   % [Nt,1]

            feat((k-1)*2*Nt + (1:2*Nt)) = single([real(v); imag(v)]);
        end

        X(:,n) = feat;
    end
end