function filepath = save_checkpoint_var(varName, varValue, dataDir)
% save_checkpoint_var
% Save one variable into dataDir as:
%   varName-yyyymmdd-HHMMSS.mat
%
% Example:
%   save_checkpoint_var('Hf_A', Hf_A, 'data')

    arguments
        varName (1,:) char
        varValue
        dataDir (1,:) char = 'data'
    end

    if ~exist(dataDir, 'dir')
        mkdir(dataDir);
    end

    ts = datestr(now, 'yyyymmdd-HHMMSS');
    filename = sprintf('%s-%s.mat', varName, ts);
    filepath = fullfile(dataDir, filename);

    % Put the variable into a struct so the saved variable name is preserved
    S = struct();
    S.(varName) = varValue;

    % Use -v7.3 for large arrays / dlnetwork safety
    save(filepath, '-struct', 'S', '-v7.3');

    fprintf('[checkpoint] saved %s -> %s\n', varName, filepath);
end