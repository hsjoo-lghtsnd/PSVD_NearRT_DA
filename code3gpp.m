% Writer: Hosung Joo (hosung.joo@postech.ac.kr)
% July 1st, 2025
% All rights reserved, AiSLab SNU

function [reconstructed, nmse, cossim] = code3gpp(H, Nt, type, L, codebook)
% H in shape of N x Nt x Nsubcarrier (required)
% N: num of samples. Multiple samples can be calculated at a single call
% Nt: num of Tx ANT. ([Nh, Nv], our case default: 10x10)
% type: 1 or 2. Default: 1
% L: Type-II only. Linear combination number.
% codebook: specified codebook, or default DFT matrices. (cell)

% Test of H shape
[N, Ntx, Nsub] = size(H);

% setting default values..
switch nargin
    case 1
        Nt = [10, 10];
        type=1;
        codebook = makecodebook1(Nt);
    case 2
        type=1;
        codebook = makecodebook1(Nt);
    case 3
        fprintf('code3gpp() is generating codebook automatically ... ');
        switch type
            case 1
                codebook = makecodebook1(Nt);
            case 2
                codebook = makecodebook2(Nt);
                L = 4;
                fprintf('setting L = %d ... ', L);
            otherwise
                error(['\ncode3gpp(): 3rd variable "type" is unknown,' ...
                    'received: %d\n'], type);
        end
        fprintf('Done\n');
    case 4
        fprintf('code3gpp() is generating codebook automatically ... ');
        switch type
            case 1
                codebook = makecodebook1(Nt);
            case 2
                codebook = makecodebook2(Nt);
                L = 4;
            otherwise
                error(['\ncode3gpp(): 3rd variable "type" is unknown,' ...
                    'received: %d\n'], type);
        end
        fprintf('Done\n');
        if type==2
            fprintf('L = %d\n', L);
        end
    case 5
        fprintf('code3gpp() is working properly ...\n');
        if type==2
            fprintf('L = %d\n', L);
        end
    otherwise
        error('code3gpp(): too many variables! Received: %d\n', nargin);
end

fprintf('N samples, Nt ANTs, Nsub taps. (N x Nt x Nsub)\n');
fprintf('Processing: %d x %d x %d H matrix, with Nt=%d x %d...\n', N, Ntx, Nsub, Nt(1), Nt(2));

if (Ntx ~= Nt(1)*Nt(2))
    error('Given input "Nt" does not match to input H size(2). Check contract.\n');
end

reconstructed = complex(zeros(N, Ntx, Nsub));
nmse = zeros(N,1);
cossim = zeros(N,1);

codebook = codebook(:);
Ncode = length(codebook);

if type==2
    [tempsize1, tempsize2] = size(codebook{1});
end

% noise_power = 1e-10;
for h_idx=1:N
    % pmi_list = zeros(1, Nsub);
    % cqi_list = zeros(1, Nsub);
    
    for f = 1:Nsub
        h_f = H(h_idx,:,f);
        h_f = h_f(:);
        best_gain = eps;

        if type==2
            bests = zeros(L,2);
        end
        for i = 1:Ncode
            w = codebook{i};
            w = w(:);
            gain = w' * h_f;
            switch type
                case 1
                    if abs(gain) > abs(best_gain)
                        best_gain = gain;
                        best_precoder = w;
                        % pmi_idx = sub2ind([Nt(1), Nt(2)], i, j);
                    end
                case 2
                    if abs(gain) > abs(bests(L,1))
                        idx = binarySearchDesc(abs(gain), abs(bests(L,1)));
                        if idx <= L
                            bests(idx+1:end, :) = bests(idx:end-1, :);
                            bests(idx,1) = gain;
                            bests(idx,2) = i;
                        end
                    end
                otherwise
                    error('unknown type: breakpoint hash 5x6x8a\n');
            end
        end
        % pmi_list(f) = pmi_idx;
        % cqi_list(f) = 10 * log10(best_gain / noise_power);  % SNR in dB
        switch type
            case 1
                reconstructed(h_idx,:,f) = best_gain * best_precoder;  % scaled reconstruction. ORG Type-I: CQI scaling
            case 2
                temp = zeros(tempsize1,tempsize2);
                for i=1:L
                    temp = temp + (bests(i,1) * codebook{bests(i,2)});
                end
                reconstructed(h_idx,:,f) = temp;
            otherwise
                error('unknown type: breakpoint hash 4a5b6c\n');
        end
    end
    
    % NMSE
    org = H(h_idx,:,:);
    recon = reconstructed(h_idx,:,:);
    nmse(h_idx) = 20*log10(norm(org-recon,'fro')/norm(org,'fro'));
    cossim(h_idx) = real(org(:)' * recon(:))/(norm(org,'fro')*norm(recon,'fro'));

    if (rem(h_idx, floor(N/100))==0 && rand()>0.875)
        fprintf('%d%% was done\n', floor(h_idx/N*100));
    end
end

end

%% Type-I Codebook (DFT)
function codebook = makecodebook1(Nt)
fprintf('No codebook was specified. Generating default codebook for Type-I...\n');
N_h = Nt(1); N_v = Nt(2);

codebook = cell(N_h, N_v);

for i = 1:N_h
    w_h = exp(-1j * 2 * pi * (0:N_h-1)' * (i-1) / N_h) / sqrt(N_h);
    for j = 1:N_v
        w_v = exp(-1j * 2 * pi * (0:N_v-1)' * (j-1) / N_v) / sqrt(N_v);
        codebook{i,j} = kron(w_v, w_h);  % N_tx x 1 vector
    end
end

end

%% Type-II Codebook (TODO: Variant)
function codebook = makecodebook2(Nt)
fprintf('No codebook was specified. Generating default codebook for Type-II...\n');

codebook = makecodebook1(Nt);

end

function idx = binarySearchDesc(x, vec)
% performs binarySearch in a descending order
%
%   idx = binarySearchDesc(x, vec)
%
%   input:
%     x   - scalar
%     vec - descending order vector (Nx1, or 1xN)
%
%   output:
%     idx - to-be inserted index. (1 < idx < N+1)
%
%   e.g.:
%     vec = [100, 80, 60, 40, 20];
%     binarySearchDesc(60, vec)    %  3 
%     binarySearchDesc(70, vec)    %  3 
%     binarySearchDesc(10, vec)    %  6 

    n = numel(vec);
    lo = 1;
    hi = n;

    while lo <= hi
        mid = floor((lo + hi)/2);
        if x == vec(mid)
            idx = mid;
            return;
        elseif x < vec(mid)
            % x in right
            lo = mid + 1;
        else
            % x in left
            hi = mid - 1;
        end
    end

    % (lo = hi+1. 1 ≤ lo ≤ n+1)
    idx = lo;
end

