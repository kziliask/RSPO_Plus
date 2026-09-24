# FedRSPO+

![FedRSPO+ method: broadcast, client calibration, RSPO+ loss computation, aggregation, and personalization.](assets/fedrspo-method.png)

Code for **[FedRSPO+: A Heterogeneity-aware Algorithm for Decision-focused Federated Learning](https://openreview.net/forum?id=zbsbqMhkWi)**, accepted at **NeurIPS 2026**.

**Konstantinos Ziliaskopoulos, Alexander Vinel, and Jiaqi Wang**

The repository includes the code to reproduce the results in the paper and our FedRSPO+ implementation.

## Setup

- Julia 1.12 (the environment is pinned in `Manifest.toml`).
- Gurobi installed with a license available to Julia.
- For Warcraft only, download the one-skin dataset from the [original benchmark repository](https://github.com/martius-lab/blackbox-differentiation-combinatorial-solvers) and place its `12x12` folder at `src/data/warcraft_shortest_path_oneskin/12x12/`.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Main experiments

```bash
# Experiment 1: synthetic knapsack
julia --project=. scripts/experiments/synth_run.jl

# Experiment 2: Warcraft shortest path
julia --project=. scripts/experiments/warcraft_run.jl

# Experiment 3: PJM battery dispatch
julia --project=. scripts/experiments/pjm_sweep_run.jl
```

## PJM ablations

LOCO excludes each target client from federated training and validation, then compares the donor model, target personalization, and local-only training with 3, 7, 14, 30, and 60 labeled target days:

```bash
julia --project=. scripts/experiments/pjm_cold_start_run.jl
```

Fixed-demand experiments assign the same dispatch demand to every PJM client and compare MSE, SPO+, and the FedRSPO+ components:

```bash
for demand in 4 6 8 10 12; do
  julia --project=. scripts/experiments/pjm_ablation_run.jl \
    --fixed-demand "$demand" --outdir "results/experiment3/pjm_fixed_demand_$demand"
done
```

## Baselines

Interp-DFFL trains local and federated SPO+ models, then selects per-client prediction interpolation weights using SPO+ or MSE calibration loss. The synthetic experiment uses an 80/20 fit/calibration split within each client's training data. PJM uses its chronological validation split.

```bash
julia --project=. scripts/experiments/synth_interp_dffl_run.jl
julia --project=. scripts/experiments/pjm_interp_dffl_run.jl
```

OptNet is implemented as an implicitly differentiated quadratic-program layer using DiffOpt and Gurobi. It tunes the quadratic regularization on validation data and evaluates decisions with the original linear optimization objective.

```bash
# Run synthetic knapsack and PJM, or select either dataset
julia --project=. scripts/experiments/id_qp_run.jl --datasets synthetic,pjm
```

## Citation

```bibtex
@inproceedings{ziliaskopoulos2026fedrspo,
  author    = {Konstantinos Ziliaskopoulos and Alexander Vinel and Jiaqi Wang},
  title     = {{FedRSPO+}: A Heterogeneity-aware Algorithm for Decision-focused Federated Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026},
  note      = {Accepted at NeurIPS 2026; proceedings forthcoming},
  url       = {https://openreview.net/forum?id=zbsbqMhkWi}
}
```
