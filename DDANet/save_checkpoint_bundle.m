function filepath = save_checkpoint_bundle(bundleName, dataDir, varargin)
% save_checkpoint_bundle
% Save multiple named variables into one MAT file:
%   bundleName-yyyymmdd-HHMMSS.mat
%
% Usage:
%   save_checkpoint_bundle('pretrained_dda', 'data', ...
%       'encoderNet0', encoderNet0, ...
%       'decoderNet0', decoderNet0, ...
%       'preHist', preHist);
%
% Inputs must come as name-value pairs.

    arguments
        bundleName (1,:) char
        dataDir (1,:) char = 'data'
    end
    arguments (Repeating)
        varargin
    end

    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end

    if mod(numel(varargin), 2) ~= 0
        error('Inputs after dataDir must be name-value pairs.');
    end

    S = struct();
    for i = 1:2:numel(varargin)
        name = varargin{i};
        value = varargin{i+1};
        if ~(ischar(name) || isstring(name))
            error('Variable names must be char or string.');
        end
        S.(char(name)) = value;
    end

    ts = datestr(now, 'yyyymmdd-HHMMSS');
    filename = sprintf('%s-%s.mat', bundleName, ts);
    filepath = fullfile(dataDir, filename);

    save(filepath, '-struct', 'S', '-v7.3');
    fprintf('[checkpoint] saved bundle %s -> %s\n', bundleName, filepath);
end