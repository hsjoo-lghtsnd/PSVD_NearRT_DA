
clear; clc;

E=4
%%

Ntrain = 100;


if E==2
    rho0 = 0.8;
    kappa0 = 0.01;
elseif E==4
    rho0 = 0.9;
    kappa0 = 0.005;
end
T = 30;
p = 0.5;
Btot = 500;

rawDataDir = 'raw_data';
resultDir = 'result';
if ~exist(resultDir, 'dir')
    mkdir(resultDir);
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

scenario_cases = 5;
maxIter = 100;
PROXY_TPR = cell(scenario_cases, scenario_cases, maxIter);
PROXY_FPR = cell(scenario_cases, scenario_cases, maxIter);
NMSE_TPR = cell(scenario_cases, scenario_cases, maxIter);
NMSE_FPR = cell(scenario_cases, scenario_cases, maxIter);
for i1 = 1:scenario_cases
    fprintf('\n[%d/5]\n', i1);

    if E==2
        name = strrep(E2_scenarios{i1}, '-', '_');
    elseif E==4
        name = E4_scenarios{i1};
    end

    htFile   = fullfile(rawDataDir, sprintf('Ht_%s.mat',   name));
    if exist(htFile, 'file')
        load(htFile, 'Ht');
    else
        fprintf('No file exists: %s\n', htFile);
        break;
    end

    Ntot = size(Ht,1);
    Ht_E = reshape(Ht, Ntot, []);

    for i2=1:scenario_cases
        if i1==i2
            continue;
        end

        if E==2
            name2 = strrep(E2_scenarios{i2}, '-', '_');
        elseif E==4
            name2 = E4_scenarios{i2};
        end
        fprintf("\n[Evaluation on E -> E': %s -> %s]\n", name, name2);

        htFile2   = fullfile(rawDataDir, sprintf('Ht_%s.mat',   name2));
        if exist(htFile2, 'file')
            load(htFile2, 'Ht');
        else
            fprintf('No file exists: %s\n', htFile2);
            break;
        end
        Ntot2 = size(Ht,1);
        Ht_Eprime = reshape(Ht, Ntot2, []);

        for iter = 1:maxIter
            if rem(iter,20)==0
                fprintf('(%d, %d): iter=%d\n', i1, i2, iter);
            end
            idx = randperm(Ntot, Ntot);
            idx2 = randperm(Ntot2, Ntrain);

            Ht_E = Ht_E(idx, :);

            Htrain = Ht_E(1:Ntrain, :);

            Htest_FPR = Ht_E(Ntrain+1:end, :);
            Htest_TPR = Ht_Eprime(idx2, :);

            [Vs, sigmaL, V, ~] = psvd_codebook(Htrain, kappa0, rho0, T);
            bSeq = bit_alloc(sigmaL, Btot, p);
            decoder = pinv(Vs);

            % FPR:
            Z = Htest_FPR*Vs;
            proxy_FPR = sum(abs(Z).^2,2)./sum(abs(Htest_FPR).^2,2);
            proxy_FPR = 10*log10(1-proxy_FPR);
            PROXY_FPR{i1, i2, iter} = proxy_FPR;

            Zq = quantize_wrapper(Z, bSeq, sigmaL);
            tildeH_FPR = Zq*decoder;
            [~, NMSElist_FPR, ~] = my_print_error(Htest_FPR, tildeH_FPR);
            NMSE_FPR{i1, i2, iter} = NMSElist_FPR;

            % TPR:
            Z = Htest_TPR*Vs;
            proxy_TPR = sum(abs(Z).^2,2)./sum(abs(Htest_TPR).^2,2);
            proxy_TPR = 10*log10(1-proxy_TPR);
            PROXY_TPR{i1, i2, iter} = proxy_TPR;

            Zq = quantize_wrapper(Z, bSeq, sigmaL);
            tildeH_TPR = Zq*decoder;
            [~, NMSElist_TPR, ~] = my_print_error(Htest_TPR, tildeH_TPR);
            NMSE_TPR{i1, i2, iter} = NMSElist_TPR;
        end
    end

    

    
end

save_checkpoint_bundle(sprintf('TPRFPR_E%d', E), resultDir, ...
    'PROXY_FPR', PROXY_FPR, ...
    'PROXY_TPR', PROXY_TPR, ...
    'NMSE_FPR', NMSE_FPR, ...
    'NMSE_TPR', NMSE_TPR);