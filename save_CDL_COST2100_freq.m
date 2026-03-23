
Ns = 3000;

E_opts = struct;
E_opts.seed = 12345;

dataDir = 'raw_data';
if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

E2_scenarios = {...
    'CDL-A'
    'CDL-B'
    'CDL-C'
    'CDL-D'
    'CDL-E'
    };

E4_scenarios = {...
    'Indoor_CloselySpacedUser_2_6GHz'; ...
    'IndoorHall_5GHz'; ...
    'SemiUrban_CloselySpacedUser_2_6GHz'; ...
    'SemiUrban_300MHz'; ...
    'SemiUrban_VLA_2_6GHz'
    };

for i = 1:5
    fprintf('\n[%d/5]\n', i);

    % ===== E2: CDL =====
    name = strrep(E2_scenarios{i}, '-', '_');

    % htFile   = fullfile(dataDir, sprintf('Ht_%s.mat',   name));
    horgFile = fullfile(dataDir, sprintf('Horg_%s.mat', name));

    % if exist(htFile, 'file') && exist(horgFile, 'file')
    if exist(horgFile, 'file')
        fprintf('E2 %s already exists. Skipping.\n', E2_scenarios{i});
    else
        fprintf('Generating E2 %s ...\n', E2_scenarios{i});
        % [Ht, ~] = load_environment(2, Ns, i, E_opts);
        % N = size(Ht,1);
        % save(htFile,   'Ht', 'N',   '-v7.3');

        [~, Horg] = load_environment(2, Ns, i, E_opts);
        N = size(Horg,1);
        save(horgFile, 'Horg', 'N', '-v7.3');
        fprintf('Saved E2 %s\n', E2_scenarios{i});
    end

    clear Ht; clear Horg;
    % ===== E4: COST2100 =====
    % htFile   = fullfile(dataDir, sprintf('Ht_%s.mat',   E4_scenarios{i}));
    horgFile = fullfile(dataDir, sprintf('Horg_%s.mat', E4_scenarios{i}));

    if exist(horgFile, 'file')
        fprintf('E4 %s already exists. Skipping.\n', E4_scenarios{i});
    else
        fprintf('Generating E4 %s ...\n', E4_scenarios{i});
        % [Ht, ~] = load_environment(4, Ns, i, E_opts);
        % N = size(Ht,1);
        % save(htFile,   'Ht', 'N',   '-v7.3');
        
        [~, Horg] = load_environment(4, Ns, i, E_opts);
        N = size(Horg,1);
        save(horgFile, 'Horg', 'N', '-v7.3');
        fprintf('Saved E4 %s\n', E4_scenarios{i});
    end
    clear Ht; clear Horg;
end

