# fal-task

## Two Moons diffusion baseline

This repository contains a minimal test bed for diffusion models on 2D data. The baseline learns the "Two Moons" distribution starting from a standard 2D Gaussian, and uses an Euler–Maruyama sampler for generation.

### Model
- **Score network**: `ScoreNet` is a small MLP conditioned on time via sinusoidal embeddings (`diffusers` `Timesteps` + `TimestepEmbedding`).
- **Scheduler (training)**: Discrete DDPM scheduler (`DDPMScheduler`, linear betas, 1000 steps). Training uses the standard ε-prediction objective: the model predicts the added noise and minimizes MSE(ε̂, ε).
- **Sampling (Euler)**: Inference integrates the reverse VP dynamics with Euler–Maruyama. We convert ε̂ → score using `alphas_cumprod` from the DDPM scheduler and approximate β(t) from the discrete schedule.

### Training
- Dataset: `sklearn` two moons (50k points).
- Optimizer: AdamW, optional gradient clipping.
- Defaults: `--train-steps 5000`, batch size 512.
- A checkpoint is saved to `artifacts/checkpoints/baseline.pt`.

### Sampling (Euler)
Starting from `N(0, I)`, we integrate from `t=1 → 0` using Euler–Maruyama on the reverse SDE. The drift uses the score derived from the model’s ε prediction; β(t) and σ(t) are computed from the DDPM schedule. Defaults: `--sample-steps 1000`, `--num-samples 5000`.

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
