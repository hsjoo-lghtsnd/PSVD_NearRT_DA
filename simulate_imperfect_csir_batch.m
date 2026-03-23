function [H, sigma2_csi] = simulate_imperfect_csir_batch(Horg, snrCsi_dB, mode)
%SIMULATE_IMPERFECT_CSIR_BATCH
% H = Horg + N, N ~ CN(0, sigma2_csi)
% Horg: [Ntot, Nt, Nr, Nsub]
%
% SNR definition (recommended, scale-safe even if Horg not normalized):
%   SNR_CSI-RS = E[ |Horg|^2 ] / sigma2_csi
%
% mode:
%   'global'         : one sigma2 over all samples/subcarriers/antennas
%   'perSample'      : sigma2 per sample (averaged over Nt,Nr,Nsub)
%   'perSampleSub'   : sigma2 per (sample, subcarrier) (averaged over Nt,Nr)

    if nargin < 3 || isempty(mode)
        mode = 'global';
    end

    snrLin = 10.^(snrCsi_dB/10);
    [Ntot, Nt, Nr, Nsub] = size(Horg);

    switch lower(mode)
        case 'global'
            sigPow = mean(abs(Horg(:)).^2);
            sigma2_csi = sigPow / snrLin;
            N = sqrt(sigma2_csi/2) * (randn(size(Horg)) + 1j*randn(size(Horg)));
            H = Horg + N;

        case 'persample'
            sigma2_csi = zeros(Ntot,1);
            H = zeros(size(Horg), 'like', Horg);
            for t = 1:Ntot
                sigPow_t = mean(abs(reshape(Horg(t,:,:,:),1,[])).^2);
                sigma2_csi(t) = sigPow_t / snrLin;
                Ntensor = sqrt(sigma2_csi(t)/2) * (randn(Nt,Nr,Nsub) + 1j*randn(Nt,Nr,Nsub));
                H(t,:,:,:) = Horg(t,:,:,:) + Ntensor;
            end

        case 'persamplesub'
            sigma2_csi = zeros(Ntot, Nsub);
            H = zeros(size(Horg), 'like', Horg);
            for t = 1:Ntot
                for k = 1:Nsub
                    Htk = squeeze(Horg(t,:,:,k)); % [Nt,Nr]
                    sigPow_tk = mean(abs(Htk(:)).^2);
                    sigma2_csi(t,k) = sigPow_tk / snrLin;
                    Ntk = sqrt(sigma2_csi(t,k)/2) * (randn(Nt,Nr) + 1j*randn(Nt,Nr));
                    H(t,:,:,k) = Horg(t,:,:,k) + Ntk;
                end
            end

        otherwise
            error("mode must be 'global', 'perSample', or 'perSampleSub'.");
    end
end