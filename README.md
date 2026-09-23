# Graph Neural Network for Particle-Level Background Correction

This project develops a graph neural network (GNN) to improve particle-level background correction in large-scale collision datasets. A conventional, data-driven correction method (Iterative Constituent Subtraction (ICS)) is first established to provide a model-independent baseline applicable to real data, after which the GNN learns to further reduce residual errors and long-tail outliers. The approach improves measurement accuracy by approximately 2.4× across 2+ TB of data. The study uses the Min Bias PbPb 0-10% data collection by the ALICE Experiment at CERN and the Monte Carlo pp data anchored to the min bias PbPb dataset.

![GNN performance comparison](GNN_ICS.png)

## Results

Key quantitative result
- ~2.4× improvement in measurement accuracy
- Reduced residual errors and long-tail outliers
- Comparison against conventional correction

## Approach

1. Classical baseline
2. Graph construction
3. Graph neural network
4. Physics-informed loss
5. Evaluation

## Why this approach?

Explain why the conventional method was developed first
and why the GNN is trained against it.

## Repository Structure

...

## Installation

...

## Running the Model

...

## Data

...

## Citation / Related Work
