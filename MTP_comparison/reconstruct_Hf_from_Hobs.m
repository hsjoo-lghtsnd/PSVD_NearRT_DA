function Hf_hat = reconstruct_Hf_from_Hobs(Hobs, scIdx, Nsub, method)
% Reconstruct full-band frequency CSI from sparse observations
%
% Input:
%   Hobs   : [Ns, Nt, Nr, Nobs]
%   scIdx  : [Nobs,1] or [1,Nobs], 1-based subcarrier indices
%   Nsub   : full active-band subcarrier count
%   method : 'zerofill', 'linear', 'pchip', 'spline'
%
% Output:
%   Hf_hat : [Ns, Nt, Nr, Nsub]

    arguments
        Hobs {mustBeNumeric}
        scIdx {mustBeNumeric}
        Nsub (1,1) {mustBeInteger, mustBePositive}
        method (1,1) string = "linear"
    end

    assert(ndims(Hobs) == 4, 'Hobs must be [Ns, Nt, Nr, Nobs].');

    [Ns, Nt, Nr, Nobs] = size(Hobs);
    scIdx = scIdx(:);
    assert(numel(scIdx) == Nobs, 'numel(scIdx) must equal size(Hobs,4).');
    assert(all(scIdx >= 1 & scIdx <= Nsub), 'scIdx out of range.');

    Hf_hat = complex(zeros(Ns, Nt, Nr, Nsub));
    xq = (1:Nsub).';

    switch lower(method)
        case 'zerofill'
            Hf_hat(:,:,:,scIdx) = Hobs;

        case {'linear','pchip','spline'}
            for s = 1:Ns
                for t = 1:Nt
                    for r = 1:Nr
                        y = squeeze(Hobs(s,t,r,:));  % [Nobs,1]

                        % complex interpolation: real/imag separately
                        y_re = interp1(scIdx, real(y), xq, char(method), 'extrap');
                        y_im = interp1(scIdx, imag(y), xq, char(method), 'extrap');

                        Hf_hat(s,t,r,:) = complex(y_re, y_im);
                    end
                end
            end

        otherwise
            error('Unsupported method. Use zerofill, linear, pchip, or spline.');
    end
end