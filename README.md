# fal-task

## Two Moons diffusion baseline

This repository contains a minimal test bed for diffusion models on 2D data. The baseline learns the "Two Moons" distribution starting from a standard 2D Gaussian, and uses an Euler–Maruyama sampler for generation.

### Model
- **Score network**: `ScoreNet` is a small MLP conditioned on time via sinusoidal embeddings (`diffusers` `Timesteps` + `TimestepEmbedding`).
- **SDE schedule**: Variance-preserving (VP) schedule with linearly increasing `beta(t)`; closed-form `alpha(t)` and `sigma(t)` are used for perturbations during training.
- **Loss**: Weighted denoising score matching. Given noised samples `x_t`, the target score is `(alpha(t) * x0 - x_t) / sigma(t)^2` and the MSE is weighted by `beta(t)`.

### Training
- Dataset: `sklearn` two moons (50k points).
- Optimizer: AdamW, optional gradient clipping.
- Defaults: `--train-steps 5000`, batch size 512.
- A checkpoint is saved to `artifacts/checkpoints/baseline.pt`.

### Sampling (Euler)
Starting from `N(0, I)`, we integrate the reverse VP-SDE from `t=1 → 0` with Euler–Maruyama using the learned score. Defaults: `--sample-steps 1000` producing `--num-samples 5000` points.

### Evaluation metrics
For quantitative validation against the target two-moons samples, the pipeline reports:
- **MMD (RBF, multi-bandwidth around median heuristic)**
- **Energy Distance**
- **Sliced Wasserstein-1** (average over random 1D projections)

### How to run
```bash
pip install -r requirements.txt
python main.py --run
```
Key outputs:
- Samples: `artifacts/outputs/samples.pt`
- Plot: `artifacts/plots/baseline_two_moons.png`
- Metrics printed to stdout

### Result
![Two Moons baseline](artifacts/plots/baseline_two_moons.png)
