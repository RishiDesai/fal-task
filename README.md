# fal-task

## Two Moons diffusion baseline

This repository contains a minimal test bed for diffusion models on 2D data. The baseline learns the "Two Moons" distribution starting from a standard 2D Gaussian, and uses Diffusers' EulerDiscreteScheduler for generation.

### Model
- **Score network**: `ScoreNet` is a small MLP conditioned on time via sinusoidal embeddings (`diffusers` `Timesteps` + `TimestepEmbedding`).
- **Scheduler (training)**: Discrete DDPM scheduler (`DDPMScheduler`, cosine betas `squaredcos_cap_v2`, 1000 steps). Training uses the standard ε-prediction objective: the model predicts the added noise and minimizes MSE(ε̂, ε).
- **Sampling**: Uses the same `DDPMScheduler` with cosine betas to match training. Inputs are scaled with `scheduler.scale_model_input`, and updates follow the discrete DDPM schedule.

### Training
- Dataset: `sklearn` two moons (50k points).
- Optimizer: AdamW, optional gradient clipping.
- Defaults: `--train-steps 5000`, batch size 512.
- A checkpoint is saved to `artifacts/checkpoints/baseline.pt`.

### Sampling
Starting from `N(0, I)`, we run Diffusers' `DDPMScheduler` (cosine betas) from `t=1 → 0` using ε-prediction. Inputs are scaled with `scheduler.scale_model_input`, and updates follow the discrete DDPM schedule. Defaults: `--sample-steps 1000`, `--num-samples 5000`.

### Evaluation metrics
For quantitative validation against the target two-moons samples, the pipeline reports:
- **MMD (RBF, multi-bandwidth around median heuristic)**
- **Energy Distance**
- **Sliced Wasserstein-1** (average over random 1D projections, L=256)

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

### Baseline (40k steps)

- **Command** (40k steps, batch size 512, lr 1e-3, 1000 sample steps, 5k samples, seed 42):
  ```bash
  python main.py --run --train-steps 40000 --batch-size 512 --lr 1e-3 --log-every 500 --sample-steps 1000 --num-samples 5000 --seed 42
  ```

- **Consolidated metrics** (lower is better):

  | Setting                               | EMA | MMD (RBF) | Energy Distance | Sliced Wasserstein-1 (L=256) |
  |---------------------------------------|-----|-----------|-----------------|-------------------------------|
  | EulerDiscrete                          | No  | 0.043403  | 0.059351        | 0.053859                      |
  | DDPM (this repo)                       | No  | 0.036139  | 0.051655        | 0.043603                      |
  | DDPM + EMA (40k steps)                 | Yes | 0.000000  | 0.020453        | 0.016328                      |
  | DDPM + EMA + Cosine betas (40k steps)  | Yes | 0.000000  | 0.019954        | 0.015846                      |

- **Artifacts**:
  - Comparison plot: `artifacts/plots/baseline_two_moons.png`
  - Generated-only plot: `artifacts/plots/baseline_two_moons_generated.png`

### Larger MLP experiment

- **Description**: Same setup as baseline, but with a larger MLP in `ScoreNet`.
- **Metrics** (lower is better):

  | Setting        | MMD (RBF) | Energy Distance | Sliced Wasserstein-1 (L=256) |
  |----------------|-----------|-----------------|-------------------------------|
  | Larger MLP     | 0.000000  | 0.017666        | 0.014260                      |

- **Artifact**:
  - Generated-only plot: `artifacts/plots/baseline_two_moons_generated_largerMLP.png`

### EMA (optional)
- **What it does**: Tracks an exponential moving average (EMA) of weights during training and optionally uses EMA weights for sampling.
- **How to enable**:
  - Add `--train-use-ema` to enable EMA tracking during training (decay via `--ema-decay`, default `0.999`).
  - Add `--sample-use-ema` to prefer EMA weights when loading for sampling.
- **Note**: EMA typically improves sample quality and distance metrics; exact numbers may vary.

#### Example (40k steps, EMA for training and sampling)
- **Command**:
  ```bash
  python main.py --run --train-steps 40000 --batch-size 512 --lr 1e-3 --log-every 500 --sample-steps 1000 --num-samples 5000 --seed 42 --train-use-ema --sample-use-ema
  ```
- See consolidated metrics table above.

#### Cosine beta schedule (40k steps, EMA)
- **Command**:
  ```bash
  python main.py --run --train-steps 40000 --batch-size 512 --lr 1e-3 --log-every 500 --sample-steps 1000 --num-samples 5000 --seed 42 --train-use-ema --sample-use-ema
  ```
- See consolidated metrics table above.
  
Using `DDPMScheduler` with cosine betas (`squaredcos_cap_v2`) for both training and sampling.
