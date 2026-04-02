# Optimizers Overview

This project provides multiple optimizers with different trade-offs between latency and reward quality. Below is a concise, heuristic-level description of each optimizer as implemented.

## Core Optimizers

- **GA (Genetic Algorithm)**: Searches over PRB allocations across the prediction horizon using evolutionary operators (selection, crossover, mutation, repair). Uses reward as the fitness function. Warm-started with fast heuristics (see “Warm-start” below).
- **PSO (Particle Swarm Optimization)**: Searches over allocation weights across the prediction horizon using particle swarm dynamics. Uses reward as the fitness function. Warm-started with fast heuristics (see “Warm-start” below).
- **Greedy**: Iteratively assigns one PRB at a time to the user that gives the highest immediate reward gain. Stops early if adding a PRB no longer improves reward. Higher reward quality, higher latency.

## Baselines / Fast Heuristics

- **Static**: Equal PRB split across all users (no tier weighting). Very fast, no optimization.
- **Average (Weighted Fair Share)**: PRBs split proportional to user tier weights. Very fast, no optimization.
- **Random**: PRBs split proportional to random weights (normalized). Very fast, used as a baseline.
- **Tier Quota** (`tier_quota`): Split PRBs across tiers based on (tier weight × tier count), then split each tier’s quota equally among its users. Fast and stable; preserves tier fairness.
- **Deficit Aware** (`deficit_aware`): Estimates PRBs required per user to meet target rate using SINR at current time. Allocates PRBs to tiers proportional to total “need,” then within tier proportional to need. Fast, focuses on satisfaction deficit without iterative search.
- **Top-K Priority** (`topk_priority`): Computes SINR per user (current time only). Allocates more PRBs to users with the worst SINR (poorest channel quality). Fast, prioritizes difficult users to raise satisfaction.
- **Target-Rate Proportional** (`target_rate`): Estimates PRBs needed per user to meet their target rate using SINR. Allocates PRBs proportional to these required PRBs. Fast, aligns allocation with rate targets.
- **Hybrid Average + Deficit** (`hybrid_avg_deficit`): Starts from Average (weighted fair share) allocation, then shifts a small fixed PRB budget toward tiers with higher satisfaction deficit. Fast, improves on pure Average while keeping low latency.

## Complexity (Big‑O, per time step)

Notation:
- `U` = number of users
- `H` = horizon
- `M` = PRBs
- `P` = PSO particles
- `I` = PSO iterations
- `G` = GA generations
- `Pop` = GA population size

- **GA**: `O(G * Pop * H * (U + cost_reward))` (fitness evaluated per individual over horizon).
- **PSO**: `O(I * P * H * (U + cost_reward))` (fitness evaluated per particle over horizon).
- **Greedy**: `O(M * U * cost_reward)` (evaluates marginal gain for each PRB assignment).

- **Static**: `O(U)`
- **Average**: `O(U)`
- **Random**: `O(U)`
- **Tier Quota**: `O(U)` (group + split)
- **Deficit Aware**: `O(U)` (SINR/need estimate + split)
- **Top‑K Priority**: `O(U log U)` (sorting by SINR)
- **Target‑Rate Proportional**: `O(U)` (need estimate + split)
- **Hybrid Average + Deficit**: `O(U)` (average + deficit shift)

`cost_reward` depends on reward computation (SINR + satisfaction/penalty math).

## Warm-start (GA / PSO)

GA and PSO are warm-started using the fast heuristics above (excluding Greedy). A set of heuristic allocations is generated for `h=0`, then those allocations seed the first population members (GA) or particles (PSO). The remaining population/particles are randomly initialized. Warm-start improves convergence without significantly increasing latency.

## How to Run

```bash
python3 gNodeB/main.py --optimizer <name>
```

Supported names:
- `ga`
- `pso`
- `greedy`
- `static`
- `average`
- `tier_quota`
- `deficit_aware`
- `topk_priority`
- `target_rate`
- `hybrid_avg_deficit`
- `random`
