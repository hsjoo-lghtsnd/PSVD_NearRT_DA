function Zq = quantize_wrapper(Z, bSeq, sigmaL)
Ns = size(Z,1);
Zq = complex(zeros(size(Z)));

for i=1:Ns
    z = Z(i,:);
    [zq, ~, ~] = quantize_codeword_minifloat(z, bSeq, sigmaL, 'verbose', false);
    Zq(i,:) = zq;
end
end