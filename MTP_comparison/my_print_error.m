function [MSE, NMSE, additional_data] = my_print_error(org, recon)
remaining = org-recon;
fprintf('residual F-norm error: %f\n', norm(remaining,'fro'));
fprintf('residual vs original, matrix-wise NMSE: %f dB\n', 20*log10(norm(remaining,'fro')/norm(org,'fro')));

sample_size = size(org,1);

MSE = zeros(sample_size,1);
NMSE = zeros(sample_size,1);

% Use F-norm to calc MSE, NMSE
for i=1:sample_size
MSE(i) = norm(remaining(i,:), 'fro');
NMSE(i) = norm(remaining(i,:), 'fro')/norm(org(i,:), 'fro');
end

% compensate F-norm into Squares
MSE = MSE.*MSE;
NMSE = NMSE.*NMSE;

% Additional Prints
additional_data = cell(3,1);
additional_data{1} = mean(MSE);
additional_data{2} = 10*log10(mean(NMSE));
additional_data{3} = 10*log10(max(NMSE));

fprintf('mean MSE: %f\n', additional_data{1});
fprintf('mean NMSE: %f dB\n', additional_data{2});
fprintf('max NMSE: %f dB\n', additional_data{3});


end
