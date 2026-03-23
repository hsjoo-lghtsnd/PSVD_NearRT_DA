function [zq, plan, stats] = quantize_codeword_minifloat(z, bSeq, sigmaL, varargin)
% quantize_codeword_minifloat
% Custom low-bit floating-point quantization for a complex codeword z.
%
% Input:
%   z    : [L,1] or [1,L] complex vector
%   bSeq : [L,1] or [1,L] even integer bits for each complex symbol z_i
%   p    : hyperparameter used upstream for bit allocation
%
% Name-Value options:
%   'Verbose'      : true/false (default true)
%   'UseSubnormal' : true/false (default true)
%
% Output:
%   zq   : quantized complex vector, same size as z
%   plan : table of bit allocation per entry
%   stats: struct with error metrics
%
% Notes:
%   - Each complex z_i gets b_i bits total.
%   - Real and imaginary parts each get t_i = b_i/2 bits.
%   - Each real scalar uses a custom minifloat:
%         sign:     1 bit
%         exponent: e_i bits
%         mantissa: m_i bits
%     where e_i is chosen heuristically from t_i.
%
%   - If t_i < 3, true sign/exponent/mantissa FP is impossible.
%     In that case, a degenerate fallback is used.

    parser = inputParser;
    addParameter(parser, 'Verbose', true, @(x)islogical(x) || isnumeric(x));
    addParameter(parser, 'UseSubnormal', true, @(x)islogical(x) || isnumeric(x));
    parse(parser, varargin{:});

    verbose = logical(parser.Results.Verbose);
    useSubnormal = logical(parser.Results.UseSubnormal);

    z = z(:);
    bSeq = bSeq(:);
    sigmaL = sigmaL(:);
    
    z = z./sigmaL;

    if numel(z) ~= numel(bSeq)
        error('z and bSeq must have the same length.');
    end

    if any(mod(bSeq,2) ~= 0)
        error('All entries of bSeq must be even, since real/imag each get b_i/2 bits.');
    end

    if any(bSeq < 0)
        error('bSeq must be nonnegative.');
    end

    L = numel(z);

    % Per-entry bookkeeping
    bScalar = bSeq / 2;
    signBits = zeros(L,1);
    expBits  = zeros(L,1);
    manBits  = zeros(L,1);

    xR = real(z);
    xI = imag(z);
    qR = zeros(L,1);
    qI = zeros(L,1);

    for i = 1:L
        t = bScalar(i);

        [sBits, eBits, mBits] = allocate_minifloat_bits(t);

        signBits(i) = sBits;
        expBits(i)  = eBits;
        manBits(i)  = mBits;

        qR(i) = quantize_real_minifloat(xR(i), t, eBits, mBits, useSubnormal);
        qI(i) = quantize_real_minifloat(xI(i), t, eBits, mBits, useSubnormal);
    end

    zq = complex(qR, qI);

    absErr = abs(z - zq);
    sqErr  = abs(z - zq).^2;

    denom = sum(abs(z).^2);
    if denom > 0
        nmse = sum(sqErr) / denom;
        nmse_dB = 10*log10(nmse);
    else
        nmse = NaN;
        nmse_dB = NaN;
    end

    mse = mean(sqErr);
    rmse = sqrt(mse);
    maxAbsErr = max(absErr);

    % Per-entry relative error; avoid division by zero
    relErr = zeros(L,1);
    nzMask = abs(z) > 0;
    relErr(nzMask) = absErr(nzMask) ./ abs(z(nzMask));
    relErr(~nzMask) = 0;

    plan = table((1:L).', bSeq, bScalar, signBits, expBits, manBits, ...
        'VariableNames', {'idx','bComplex','bScalar','signBits','expBits','mantBits'});

    stats = struct();
    stats.absErr = absErr;
    stats.relErr = relErr;
    stats.sqErr = sqErr;
    stats.mse = mse;
    stats.rmse = rmse;
    stats.nmse = nmse;
    stats.nmse_dB = nmse_dB;
    stats.maxAbsErr = maxAbsErr;
    stats.z = z;
    stats.zq = zq;

    if verbose
        fprintf('\n=== Custom Minifloat Quantization Summary ===\n');
        fprintf('Length L                 : %d\n', L);
        fprintf('Mean b_i (complex)       : %.4f bits\n', mean(bSeq));
        fprintf('Mean bits per real scalar: %.4f bits\n', mean(bScalar));
        fprintf('MSE                      : %.6e\n', mse);
        fprintf('RMSE                     : %.6e\n', rmse);
        fprintf('NMSE                     : %.6e\n', nmse);
        fprintf('NMSE (dB)                : %.4f dB\n', nmse_dB);
        fprintf('Max |error|              : %.6e\n', maxAbsErr);

        fprintf('\nPer-entry results:\n');
        fprintf(' idx | b_i | split(real)=1/e/m | z_i(original)                 | z_i(quantized)                | |err|       | rel.err\n');
        fprintf('-----+-----+-------------------+-------------------------------+-------------------------------+------------+---------\n');
        for i = 1:L
            fprintf('%4d | %3d | %d/%d/%d             | % .6e%+.6ei | % .6e%+.6ei | %.6e | %.6e\n', ...
                i, bSeq(i), signBits(i), expBits(i), manBits(i), ...
                real(z(i)), imag(z(i)), real(zq(i)), imag(zq(i)), ...
                absErr(i), relErr(i));
        end
    end
    zq = zq .* sigmaL;
end

function [sBits, eBits, mBits] = allocate_minifloat_bits(t)
% Allocate sign/exponent/mantissa bits for one real scalar
% given total scalar bit budget t.
%
% Output:
%   sBits = sign bits
%   eBits = exponent bits
%   mBits = mantissa bits

    if t <= 0
        sBits = 0; eBits = 0; mBits = 0;
        return;
    end

    % Sign is always 1 when any bit exists
    sBits = 1;

    if t == 1
        % Degenerate case
        eBits = 0;
        mBits = 0;
        return;
    elseif t == 2
        % Still degenerate; very coarse
        eBits = 1;
        mBits = 0;
        return;
    end

    % For t >= 3: valid custom minifloat
    eBits = round(log2(t));
    eBits = max(1, min(t-2, eBits));
    mBits = t - sBits - eBits;
end

function q = quantize_real_minifloat(x, t, eBits, mBits, useSubnormal)
% Quantize a real scalar x using a custom minifloat format.
%
% t      : total bits for this real scalar
% eBits  : exponent bits
% mBits  : mantissa bits
%
% This function uses:
%   value = sign * 2^E * (1 + frac), for normalized numbers
%
% A simple biased exponent is used.
% Very small t are handled by fallback logic.

    if t <= 0
        q = 0;
        return;
    end

    if x == 0
        q = 0;
        return;
    end

    s = sign(x);
    a = abs(x);

    % Degenerate fallback cases
    if t == 1
        % No meaningful magnitude info
        q = 0;
        return;
    elseif t == 2
        % sign + exponent-only style coarse quantization
        % Quantize magnitude to nearest power of two with a 1-bit exponent range
        q = s * 2^round(log2(a));
        return;
    end

    % For eBits >= 1, use a minifloat-like quantizer
    bias = 2^(eBits-1) - 1;
    Emin = -bias;
    Emax = (2^eBits - 1) - bias;

    E = floor(log2(a));

    % Overflow: saturate to max finite
    if E > Emax
        fracMax = 1 - 2^(-mBits);   % largest mantissa below 1
        q = s * 2^Emax * (1 + fracMax);
        return;
    end

    % Underflow/subnormal region
    if E < Emin
        if useSubnormal
            % Smallest grid step in subnormal region
            step = 2^(Emin - mBits);
            q = s * round(a / step) * step;
        else
            q = 0;
        end
        return;
    end

    % Normalized number
    frac = a / 2^E - 1;    % in [0, 1)
    fracQ = round(frac * 2^mBits) / 2^mBits;

    % Carry handling
    if fracQ >= 1
        fracQ = 0;
        E = E + 1;

        if E > Emax
            fracMax = 1 - 2^(-mBits);
            q = s * 2^Emax * (1 + fracMax);
            return;
        end
    end

    q = s * 2^E * (1 + fracQ);
end