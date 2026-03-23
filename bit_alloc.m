function bSeq = bit_alloc(sigmaL, Btot, p)
% returns realized allocation on sigmaL, Btot, p.
L = length(sigmaL);
bSeq = zeros(size(sigmaL));

weight = sigmaL .^ p;
weight = weight/sum(weight);

for i=1:L
    bSeq(i) = min(2*floor(weight(i)*Btot/2), 64);
end

end
