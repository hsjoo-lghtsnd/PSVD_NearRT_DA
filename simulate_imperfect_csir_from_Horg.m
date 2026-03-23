function [H, sigma2_csi] = simulate_imperfect_csir_from_Horg(Horg, snrCsi_dB, mode)
%SIMULATE_IMPERFECT_CSIR_FROM_HORG
% Generate imperfect CSIR H^F = Horg^F + N^F with complex Gaussian estimation noise.
%
% Inputs:
%   Horg       : [Nt, Nr, Nsub] complex (true channel)
%   snrCsi_dB  : scalar, target CSI-RS SNR in dB (definition depends on mode)
%   mode       : 'global' (default) or 'perSubcarrier'
%
% Output:
%   H          : [Nt, Nr, Nsub] complex, imperfect CSIR
%   sigma2_csi : noise variance used (scalar for global, [1,Nsub] for perSubcarrier)

    if nargin < 3 || isempty(mode)
        mode = 'global';
    end

    snrLin = 10.^(snrCsi_dB/10);
    [Nt, Nr, Nsub] = size(Horg);

    switch lower(mode)
        case 'global'
            % Average channel power per complex coefficient (over all k, tx, rx)
            sigPow = mean(abs(Horg(:)).^2);          % E[|Horg|^2]
            sigma2_csi = sigPow / snrLin;            % sigma^2 so that sigPow/sigma^2 = SNR

            N = sqrt(sigma2_csi/2) * (randn(size(Horg)) + 1j*randn(size(Horg)));
            H = Horg + N;

        case 'persubcarrier'
            sigma2_csi = zeros(1, Nsub);
            H = zeros(size(Horg), 'like', Horg);
            for k = 1:Nsub
                sigPow_k = mean(mean(abs(Horg(:,:,k)).^2, 1), 2); % average over Nt,Nr
                sigma2_csi(k) = sigPow_k / snrLin;

                Nk = sqrt(sigma2_csi(k)/2) * (randn(Nt, Nr) + 1j*randn(Nt, Nr));
                H(:,:,k) = Horg(:,:,k) + Nk;
            end

        otherwise
            error("mode must be 'global' or 'perSubcarrier'.");
    end
end