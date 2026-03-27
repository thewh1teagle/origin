# Autoencoder Training Feasibility

## What's well-specified
- Latent encoder architecture — exact layer shapes, ConvNeXt config
- Latent decoder dilation schedule [1,2,4,1,2,4,1,1,1,1]
- Discriminator architectures (Table 7 gives exact MRD conv sizes)
- Loss coefficients (λ_recon=45, λ_adv=1, λ_fm=0.1)
- Training hyperparameters (lr=2e-4, batch=28, 1.5M steps, 4× RTX 4090)

## Missing / unclear details
- **Mel config** — fmin/fmax not stated. Standard is fmin=0 or 80Hz, fmax=8000Hz or Nyquist. 228 bands is unusual (most use 80 or 128), so wrong settings will hurt quality noticeably.
- **LR scheduler** — not mentioned for autoencoder (only TTL has one)
- **Normalization inside ConvNeXt** — LayerNorm vs GroupNorm, exact placement
- **Training data** — 11,167h includes internal data. Appendix F (public subset list) is cut off in the paper excerpt. You won't match their data scale.

## Biggest practical challenge
**Compute.** 1.5M steps on 4× RTX 4090 is weeks of training. Vocoders often converge well by 500k steps so you can likely get good results earlier, but it's still heavy.

## Bottom line
**Doable.** Architecture is fully reproducible. Open source base (Vocos + BigVGAN v2 discriminators) covers most of the code. Main risks are guessing the mel config and having less training data.
