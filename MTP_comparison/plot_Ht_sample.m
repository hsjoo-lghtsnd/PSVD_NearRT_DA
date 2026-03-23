function plot_Ht_sample(Ht_list, sampleIdx, txIdx, rxIdx, varargin)
% Plot one or more delay-domain CSI traces on the same axes.
%
% Input:
%   Ht_list   : either
%               (a) one tensor [Ns, Nt, Nr, Ntap], or
%               (b) cell array of tensors
%   sampleIdx : sample index in [1, Ns]
%   txIdx     : transmit antenna index
%   rxIdx     : receive antenna index
%
% Name-Value:
%   'Mode'       : 'power' (default), 'db', or 'magnitude'
%   'PlotType'   : 'stem' (default) or 'line'
%   'Labels'     : labels for legend
%   'LineWidth'  : line width (default: 1.5)
%   'TitleText'  : custom title text
%
% Behavior:
%   - Multiple tensors are overlaid using hold on.
%   - For 'power': plots abs(h)^2
%   - For 'db': plots 10*log10(abs(h)^2 + eps)
%   - For 'magnitude': plots abs(h)

    p = inputParser;
    addParameter(p, 'Mode', 'power');
    addParameter(p, 'PlotType', 'stem');
    addParameter(p, 'Labels', []);
    addParameter(p, 'LineWidth', 1.5);
    addParameter(p, 'TitleText', '');
    parse(p, varargin{:});
    opt = p.Results;

    if ~iscell(Ht_list)
        Ht_list = {Ht_list};
    end

    nCurves = numel(Ht_list);

    % Labels
    if isempty(opt.Labels)
        labels = arrayfun(@(k) sprintf('Ht{%d}', k), 1:nCurves, 'UniformOutput', false);
    else
        labels = opt.Labels;
        if isstring(labels), labels = cellstr(labels); end
        assert(numel(labels) == nCurves, 'Number of labels must match number of tensors.');
    end

    figure;
    hold on;
    grid on;

    for k = 1:nCurves
        Ht = Ht_list{k};
        assert(ndims(Ht) == 4, 'Each Ht must be [Ns, Nt, Nr, Ntap].');

        [Ns, Nt, Nr, Ntap] = size(Ht);
        assert(sampleIdx >= 1 && sampleIdx <= Ns, 'sampleIdx out of range.');
        assert(txIdx >= 1 && txIdx <= Nt, 'txIdx out of range.');
        assert(rxIdx >= 1 && rxIdx <= Nr, 'rxIdx out of range.');

        h = squeeze(Ht(sampleIdx, txIdx, rxIdx, :));  % [Ntap,1]

        switch lower(opt.Mode)
            case 'power'
                y = abs(h).^2;
                yLabelText = 'Power';
            case 'db'
                y = 10*log10(abs(h).^2 + eps);
                yLabelText = 'Power (dB)';
            case 'magnitude'
                y = abs(h);
                yLabelText = 'Magnitude';
            otherwise
                error('Mode must be ''power'', ''db'', or ''magnitude''.');
        end

        x = 1:Ntap;

        switch lower(opt.PlotType)
            case 'stem'
                s = stem(x, y, 'DisplayName', labels{k});
                s.LineWidth = opt.LineWidth;
            case 'line'
                plot(x, y, 'LineWidth', opt.LineWidth, 'DisplayName', labels{k});
            otherwise
                error('PlotType must be ''stem'' or ''line''.');
        end
    end

    xlabel('Tap index');
    ylabel(yLabelText);

    if isempty(opt.TitleText)
        title(sprintf('H_t comparison: sample=%d, tx=%d, rx=%d', sampleIdx, txIdx, rxIdx));
    else
        title(opt.TitleText);
    end

    legend('Location', 'best');
    hold off;
end