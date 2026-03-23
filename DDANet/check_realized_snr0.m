function SNR0_dB_check = check_realized_snr0(HorgSet, Nlayer, Ptot, N0)
    [Nt, Nr, Nsub, ~] = size(HorgSet);

    Hpow = squeeze(sum(abs(HorgSet).^2, [1 2]));
    EH = mean(Hpow(:));

    eta = Ptot / (Nsub * Nlayer);
    SNR0 = (eta / N0) * (EH / (Nt * Nr));
    SNR0_dB_check = 10 * log10(SNR0);
end