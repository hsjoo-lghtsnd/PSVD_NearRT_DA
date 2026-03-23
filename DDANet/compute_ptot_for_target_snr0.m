function Ptot = compute_ptot_for_target_snr0(HorgSet, Nlayer, N0, SNR0_dB)
% compute_ptot_for_target_snr0
%
% HorgSet : [Nt, Nr, Nsub, Nsamples] complex
% Nlayer  : number of spatial layers
% N0      : noise variance per receive antenna
% SNR0_dB : target normalized SNR in dB
%
% Uses:
%   SNR0 = (eta/N0) * E[||H[k]||_F^2] / (Nt*Nr)
%   eta  = Ptot / (Nsub*Nlayer)
%
% Therefore:
%   Ptot = SNR0 * N0 * Nsub * Nlayer * Nt*Nr / E[||H[k]||_F^2]

    arguments
        HorgSet {mustBeNumeric}
        Nlayer (1,1) {mustBeInteger, mustBePositive}
        N0 (1,1) double {mustBePositive}
        SNR0_dB (1,1) double
    end

    [Nt, Nr, Nsub, ~] = size(HorgSet);

    SNR0 = 10^(SNR0_dB/10);

    % Frobenius power per subcarrier/sample: [Nsub, Nsamples]
    Hpow = squeeze(sum(abs(HorgSet).^2, [1 2]));   % sum over Nt,Nr

    % E_{t,k}[ ||H[k]||_F^2 ]
    EH = mean(Hpow(:));

    Ptot = SNR0 * N0 * Nsub * Nlayer * (Nt * Nr) / EH;
end