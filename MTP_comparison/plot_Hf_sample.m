function plot_Hf_sample(Hf_list, sampleIdx, txIdx, rxIdx, varargin)
% Plot one or more frequency-domain CSI traces on the same axes.
% Additionally, plot phase responses in a separate figure and display
% maximum absolute phase differences.
%
% Input:
%   Hf_list   : either one tensor [Ns, Nt, Nr, Nsub] or a cell array of them
%   sampleIdx : sample index
%   txIdx     : transmit antenna index
%   rxIdx     : receive antenna index
%
% Name-Value:
%   'Labels'      : labels for legend
%   'Scale'       : 'db' (default), 'linear', or 'phase'
%                   - Controls the FIRST figure (magnitude/phase-as-main plot)
%                   - SECOND figure always shows angle
%   'LineWidth'   : line width
%   'TitleText'   : custom title
%   'ScIdx'       : subcarrier indices for sparse observations
%   'Hobs'        : sparse observed values, shape [Ns, Nt, Nr, Nobs]
%   'ObsLabel'    : legend label for Hobs stem
%
% Behavior:
%   - Plots Hf curves in the original style (first figure)
%   - If both ScIdx and Hobs are provided, overlays stem(scIdx, Hobs(...))
%   - Adds a second figure for angle(Hf)
%   - Displays maximum absolute wrapped phase difference against Hf_list{1}

    p = inputParser;
    addParameter(p, 'Labels', []);
    addParameter(p, 'Scale', 'db');
    addParameter(p, 'LineWidth', 1.5);
    addParameter(p, 'TitleText', '');
    addParameter(p, 'ScIdx', []);
    addParameter(p, 'Hobs', []);
    addParameter(p, 'ObsLabel', 'Observed H_f');
    parse(p, varargin{:});
    opt = p.Results;

    if ~iscell(Hf_list)
        Hf_list = {Hf_list};
    end

    nCurves = numel(Hf_list);

    if isempty(opt.Labels)
        labels = arrayfun(@(k) sprintf('Hf{%d}', k), 1:nCurves, 'UniformOutput', false);
    else
        labels = opt.Labels;
        if isstring(labels), labels = cellstr(labels); end
        assert(numel(labels) == nCurves, 'Number of labels must match number of tensors.');
    end

    % Precompute selected traces
    hCell = cell(1, nCurves);
    angleCell = cell(1, nCurves);
    magPlotCell = cell(1, nCurves);

    for k = 1:nCurves
        Hf = Hf_list{k};
        assert(ndims(Hf) == 4, 'Each Hf must be [Ns, Nt, Nr, Nsub].');

        [Ns, Nt, Nr, Nsub] = size(Hf);
        assert(sampleIdx >= 1 && sampleIdx <= Ns, 'sampleIdx out of range.');
        assert(txIdx >= 1 && txIdx <= Nt, 'txIdx out of range.');
        assert(rxIdx >= 1 && rxIdx <= Nr, 'rxIdx out of range.');

        h = squeeze(Hf(sampleIdx, txIdx, rxIdx, :));  % [Nsub,1]
        hCell{k} = h;
        angleCell{k} = angle(h);

        switch lower(opt.Scale)
            case 'db'
                magPlotCell{k} = 20*log10(abs(h) + eps);
                yLabelText = 'Magnitude (dB)';
            case 'linear'
                magPlotCell{k} = abs(h);
                yLabelText = 'Magnitude';
            case 'phase'
                magPlotCell{k} = angle(h);
                yLabelText = 'Phase (rad)';
            otherwise
                error('Scale must be ''db'', ''linear'', or ''phase''.');
        end
    end

    %% Figure 1: original plot behavior
    figure;
    hold on;
    grid on;

    for k = 1:nCurves
        plot(1:Nsub, magPlotCell{k}, 'LineWidth', opt.LineWidth, 'DisplayName', labels{k});
    end

    % Overlay sparse observations on first figure
    if ~isempty(opt.ScIdx) && ~isempty(opt.Hobs)
        scIdx = opt.ScIdx(:);
        Hobs = opt.Hobs;

        assert(ndims(Hobs) == 4, 'Hobs must be [Ns, Nt, Nr, Nobs].');

        [NsObs, NtObs, NrObs, Nobs] = size(Hobs);
        assert(sampleIdx >= 1 && sampleIdx <= NsObs, 'sampleIdx out of range for Hobs.');
        assert(txIdx >= 1 && txIdx <= NtObs, 'txIdx out of range for Hobs.');
        assert(rxIdx >= 1 && rxIdx <= NrObs, 'rxIdx out of range for Hobs.');
        assert(numel(scIdx) == Nobs, 'numel(scIdx) must equal size(Hobs,4).');

        hObs = squeeze(Hobs(sampleIdx, txIdx, rxIdx, :));  % [Nobs,1]

        switch lower(opt.Scale)
            case 'db'
                yObs = 20*log10(abs(hObs) + eps);
            case 'linear'
                yObs = abs(hObs);
            case 'phase'
                yObs = angle(hObs);
        end

        s = stem(scIdx, yObs, 'filled', 'DisplayName', opt.ObsLabel);
        s.LineWidth = 1.0;
    end

    xlabel('Subcarrier index');
    ylabel(yLabelText);

    if isempty(opt.TitleText)
        title(sprintf('H_f comparison: sample=%d, tx=%d, rx=%d', sampleIdx, txIdx, rxIdx));
    else
        title(opt.TitleText);
    end

    legend('Location', 'best');
    hold off;

    %% Figure 2: angle plot
    figure;
    hold on;
    grid on;

    for k = 1:nCurves
        plot(1:Nsub, angleCell{k}, 'LineWidth', opt.LineWidth, 'DisplayName', labels{k});
    end

    % Overlay sparse observations on angle figure too
    if ~isempty(opt.ScIdx) && ~isempty(opt.Hobs)
        scIdx = opt.ScIdx(:);
        Hobs = opt.Hobs;
        hObs = squeeze(Hobs(sampleIdx, txIdx, rxIdx, :));  % [Nobs,1]
        yObsAng = angle(hObs);

        s2 = stem(scIdx, yObsAng, 'filled', 'DisplayName', opt.ObsLabel);
        s2.LineWidth = 1.0;
    end

    xlabel('Subcarrier index');
    ylabel('Phase (rad)');

    if isempty(opt.TitleText)
        title(sprintf('Angle(H_f): sample=%d, tx=%d, rx=%d', sampleIdx, txIdx, rxIdx));
    else
        title([opt.TitleText ' [Angle]']);
    end

    legend('Location', 'best');
    hold off;

    %% Display max absolute wrapped phase difference
    if nCurves >= 2
        refAng = angleCell{1};

        for k = 2:nCurves
            dphi = angle(exp(1j*(angleCell{k} - refAng)));  % wrapped to [-pi, pi]
            maxAbsDphi = max(abs(dphi));
            disp(sprintf('Max |angle difference| between %s and %s: %.6f rad (%.3f deg)', ...
                labels{1}, labels{k}, maxAbsDphi, maxAbsDphi*180/pi));
        end
    end

    % Also compare Hobs phase to first curve at observed locations
    if ~isempty(opt.ScIdx) && ~isempty(opt.Hobs)
        refAngObs = angleCell{1}(scIdx);
        hObs = squeeze(opt.Hobs(sampleIdx, txIdx, rxIdx, :));
        obsAng = angle(hObs);

        dphiObs = angle(exp(1j*(obsAng - refAngObs)));
        maxAbsDphiObs = max(abs(dphiObs));

        disp(sprintf('Max |angle difference| between %s and %s at observed scIdx: %.6f rad (%.3f deg)', ...
            labels{1}, opt.ObsLabel, maxAbsDphiObs, maxAbsDphiObs*180/pi));
    end
end