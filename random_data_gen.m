% gen_save_HT(100000, 2048, "DATA_Htrainrandom.mat")
% gen_save_HT(30000, 2048, "DATA_Hvalrandom.mat")
% gen_save_HT(20000, 2048, "DATA_Htestrandom.mat")



D = 2048;
N = [100000, 30000, 20000];
spec = { "DATA_Htrainrandom.mat", N(1);
         "DATA_Hvalrandom.mat", N(2);
         "DATA_Htestrandom.mat", N(3) };
M = double(sum(N)) * double(D);
sigma = 0.5 / sqrt(2 * log(M));

seedsave_split_HT(D, sigma, spec);
