function [tildeH, QtildeH, Vs, real_Btot] = psvd_Htrain_wrapper(Htrain, Ht, kappa0, rho0, T, p, Btot)
% [tildeH, QtildeH, real_Btot] = psvd_Htrain_wrapper(Htrain, Ht, kappa0, rho0, T, p, Btot)

[Vs, sigmaL, ~, ~] = psvd_codebook(Htrain, kappa0, rho0, T);
decoder = pinv(Vs);

Ntrain = size(Htrain,1);

Z = Ht*Vs;
tildeH = Z*decoder;
sigmaL = sigmaL/sqrt(Ntrain);
bSeq = bit_alloc(sigmaL, Btot, p);

real_Btot = sum(bSeq);
% fprintf('for Btot = %d, realized total bits: %d\n', Btot, real_Btot);

Zq = quantize_wrapper(Z, bSeq, sigmaL);

QtildeH = Zq*decoder;

end