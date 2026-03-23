clear; clc;

E = 2;
resultDir = 'result';
% ---------------------------------
% Find the latest file matching:
% TPRFPR_E%d-*.mat
% ---------------------------------
pattern = sprintf('TPRFPR_E%d-*.mat', E);
dirList = dir(fullfile(resultDir, pattern));

if isempty(dirList)
    error('No file matching pattern "%s" found in "%s".', pattern, resultDir);
end

% If multiple files exist, choose the most recent one.
[~, idxLatest] = max([dirList.datenum]);
inFile = fullfile(resultDir, dirList(idxLatest).name);

fprintf('Using file: %s\n', inFile);

S = load(inFile, 'PROXY_FPR', 'PROXY_TPR');
%%
% ----------------------------
% sensitivity sweep options
% ----------------------------
opts = struct();
opts.xiGrid = -6:0.25:0;       % dB
opts.NthGrid = 1:15;           % count threshold candidates
opts.Nwin = 100;                % adaptation window length
opts.decisionRule = 'ge';      % proxy >= xi_th => suspicious
opts.aggregateMode = 'global'; % 'global' or 'pairwise_mean'

[sens, best] = threshold_sensitivity_proxy( ...
    S.PROXY_FPR, S.PROXY_TPR, opts);

disp('===== Selected operating point =====');
fprintf('xi_th = %.2f dB\n', best.xi_th);
fprintf('Nth   = %d\n', best.Nth);
fprintf('Nwin  = %d\n', best.Nwin);
fprintf('TPR   = %.6f\n', best.TPR);
fprintf('FPR   = %.6f\n', best.FPR);
fprintf('BA    = %.6f\n', best.BA);
fprintf('J     = %.6f\n', best.Youden);

save(fullfile(resultDir, sprintf('threshold_sensitivity_E%d.mat', E)), ...
    'sens', 'best', 'opts');

% ----------------------------
% plot heatmaps
% ----------------------------
plot_threshold_sensitivity(sens, opts);

%%
pairTable = evaluate_pairwise_operating_point( ...
    S.PROXY_FPR, S.PROXY_TPR, -3, 5, 20, 'ge');

disp(pairTable);
fprintf('Worst-case TPR = %.4f\n', min(pairTable.TPR));
fprintf('Worst-case FPR = %.4f\n', max(pairTable.FPR));

%%
xi_th = -3;
N_th  = 5;
W     = 20;

[tpr5, dmean5, dmed5] = analyze_proxy_tpr(PROXY_TPR, xi_th, N_th);
[fpr5, fa5, nw5]      = analyze_proxy_fpr(PROXY_FPR, xi_th, N_th, W);