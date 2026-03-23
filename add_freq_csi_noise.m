function Hnoisy = add_freq_csi_noise(Hclean, SNR_dB)
% Hclean : [Nt, Nr, Nsub, Nsamples]
% Hnoisy : same size
%
% Noise variance chosen from average signal power.

    sigPow = mean(abs(Hclean(:)).^2);
    noisePow = sigPow / (10^(SNR_dB/10));

    noise = sqrt(noisePow/2) * (randn(size(Hclean), 'like', Hclean) + 1j*randn(size(Hclean), 'like', Hclean));
    Hnoisy = Hclean + noise;
end