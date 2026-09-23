# Graph Neural Network for Particle-Level Background Correction

This project develops a machine-learning approach to improve background correction in large-scale collision datasets. The analysis progresses through three stages:
1. Area-based correction: A standard method that estimates the average background contribution from the surrounding event and subtracts it from each particle.
2. ICS correction: A more refined, particle-level correction method developed to better account for local variations in the background.
3. Graph neural network: A GNN is trained to further reduce the residual errors left by the ICS method by learning relationships between nearby particles.
The GNN approach improves measurement accuracy by approximately 2.4× across 2+ TB of data, while reducing residual errors and long-tail outliers.

The study uses the Min Bias PbPb 0-10% data collection by the ALICE Experiment at CERN and the Monte Carlo pp data anchored to the min bias PbPb dataset.

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
