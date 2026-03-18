Core entry point:
- run_ddanet_psvd_implicit_sweep.m

Main pipeline:
1. Generate CDL source/target CSI
2. Convert CSI to 832-d implicit eigenvector features
3. Pretrain DDA-Net backbone on source data
4. Finetune DDA-Net on target field samples
5. Build PSVD codebooks on the same implicit feature space
6. Evaluate spectral efficiency using reconstructed subband precoders
7. Save checkpoints under ./data

```
run_ddanet_psvd_implicit_sweep.m
├─ save_checkpoint_var.m
├─ save_checkpoint_bundle.m
├─ generate_cdl_freq_csi.m
├─ makeImplicit13SubbandInput.m
├─ createImCsiNetM832.m
├─ pretrainExistingImCsiNetForDDA.m
│  ├─ preprocessMiniBatch.m
│  ├─ modelGradientsImCsiNetM.m
│  │  ├─ negativeCosineSimilarityLossMultiSubband.m
│  │  └─ steStochasticBinarize.m
│  └─ evaluateImCsiNetM.m
│     ├─ negativeCosineSimilarityLossMultiSubband.m
│     └─ cosineSimilarityPerSampleMultiSubband.m
├─ buildPrestoredCodewordBank.m
├─ predictImCsiNetAutoencoder.m
├─ finetuneDDAImCsiNetS.m
│  ├─ preprocessMiniBatch.m
│  ├─ modelGradientsDDAImCsiNetS.m
│  │  ├─ negativeCosineSimilarityLoss.m
│  │  ├─ mmdRbfLoss.m
│  │  └─ steStochasticBinarize.m
│  └─ evaluateImCsiNetS.m
│     ├─ negativeCosineSimilarityLoss.m
│     └─ steDeterministicBinarizeForEval (local function inside file)
├─ psvd_codebook.m
├─ psvd_reconstruct_features.m
├─ batch_eval_rate_from_ddanet_outputs.m
│  ├─ make_contiguous_subband_map.m
│  └─ su_mimo_ofdm_rate_from_ddanet_output.m
│     ├─ unpack_implicit_features.m
│     └─ su_mimo_ofdm_rate_given_precoder.m
├─ avg_cosine_similarity_matrix.m
│  └─ unpack_implicit_features.m
└─ add_freq_csi_noise.m   [optional, only if usePerfectCSIR = false]
```
