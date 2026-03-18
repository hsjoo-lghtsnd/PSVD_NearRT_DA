function subband_map = make_contiguous_subband_map(Nsub, Nsband)
% Map each subcarrier to one of Nsband contiguous subbands.
% Example: Nsub=624, Nsband=13 => each subband has 48 subcarriers.

    assert(mod(Nsub, Nsband) == 0, 'Nsub must be divisible by Nsband.');
    w = Nsub / Nsband;

    subband_map = zeros(1, Nsub);
    for s = 1:Nsband
        idx = (s-1)*w + (1:w);
        subband_map(idx) = s;
    end
end