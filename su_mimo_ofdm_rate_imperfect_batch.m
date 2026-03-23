function [R_avg, Rperf_avg] = su_mimo_ofdm_rate_imperfect_batch(Horg_test, H_test, Htilde, Nlayer, Ptot, N0, opts)
% Horg, H, and tildeH in shape [Nsample, Nt, Nr, Nsub]
if nargin < 7, opts = struct(); end

Neval = size(Horg_test,1);

Horg_test = permute(Horg_test, [2,3,4,1]); % [Nt, Nr, Nsub, Nsample]
H_test = permute(H_test, [2,3,4,1]);
Htilde = permute(Htilde, [2,3,4,1]);

R_list = zeros(Neval,1);
Rperf_list = zeros(Neval,1);

for t = 1:Neval
    Horg_t   = Horg_test(:,:,:,t);    % [Nt,Nr,Nsub]
    H_t      = H_test(:,:,:,t);       % [Nt,Nr,Nsub]
    Htilde_t = Htilde(:,:,:,t);       % [Nt,Nr,Nsub]

    [R_list(t), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, H_t, Htilde_t, Nlayer, Ptot, N0, opts);
    [Rperf_list(t), ~] = su_mimo_ofdm_rate_imperfect(Horg_t, Horg_t, Horg_t, Nlayer, Ptot, N0, opts);
end

R_avg = mean(R_list(~isnan(R_list)));
Rperf_avg = mean(Rperf_list(~isnan(Rperf_list)));

exact_num_R = sum(~isnan(R_list));
exact_num_Rperf = sum(~isnan(Rperf_list));
fprintf("Average R over %d samples = %.6f bps/Hz\n", exact_num_R, R_avg);
fprintf("Average R over %d samples = %.6f bps/Hz (when perfect CSIR/CSIT)\n", exact_num_Rperf, Rperf_avg);

end
