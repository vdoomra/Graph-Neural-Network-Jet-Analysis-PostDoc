# Graph Neural Network for Particle-Level Background Correction

This project develops a graph neural network (GNN) for background subtraction in heavy-ion collisions, using data from the ALICE experiment at CERN. The GNN is trained on the output of a classical correction method rather than on Monte Carlo truth labels. As a result, it does not depend on an event-generator model, and it can in principle be trained directly on real data.

<p align="center">
  <img src="GNN_ICS.png" width="650" alt="Residual distributions for the area-based, ICS and GNN corrections">
</p>

## Motivation

In central lead–lead (Pb–Pb) collisions, jets (collimated sprays of particles from high-energy quarks and gluons) sit on top of a large, fluctuating background of thousands of soft particles. To measure a jet's momentum accurately, that background has to be subtracted.

Machine-learning corrections are usually trained on simulation, where the true jet momentum is known. The trained model then inherits the assumptions of the event generator used to produce the simulation. This model dependence is difficult to quantify, and it becomes a systematic uncertainty on the final measurement.

## Approach

The analysis progresses through three stages:

1. **Area-based correction.** This is the standard method. It estimates the average background density ρ in each event and subtracts ρ × (jet area) from the jet.
2. **ICS correction.** A particle-level method that corrects each particle individually, better accounting for local background fluctuations.
3. **Graph neural network.** The GNN is trained to reproduce the ICS correction. It then refines ICS by using correlations between neighbouring particles, which ICS does not exploit.

Because the GNN's training target comes from a data-driven method rather than from generator truth, the correction carries no generator-model dependence.

For comparison, the repository also includes a dense neural network trained directly on Monte Carlo truth labels. This is the conventional supervised approach.

| | Dense NN (`neural_network_approach.py`) | GNN (`train_gnn_ics.py`) |
|---|---|---|
| Level | Jet | Particle |
| Training target | MC truth jet pT | ICS-corrected particle pT |
| Requires truth labels | Yes | No |
| Generator-model dependence | Yes | No |

## Method

**Graph construction.** Each event is represented as a graph. Nodes are particles, with features pT, η, φ and ρ. Edges connect particle pairs within ΔR < 0.3 in the (η, φ) plane, found with a KD-tree radius search.

**Model.** Three GraphSAGE layers followed by an MLP predict the corrected pT of each particle (node-level regression).

**Physics-informed loss.** Background subtraction can only remove momentum, so a penalty is added when the predicted pT exceeds the input pT. The penalty is weighted by (1 + pT), and its strength ramps up over the first epochs. At inference the constraint is enforced exactly by clamping.

**Export.** The trained model is saved in TorchScript format so it can be loaded in the C++ analysis framework.

**Dense-NN comparison.** A fully connected network (Keras) regresses the true jet pT from reconstructed jet features, with an L2-regularised loss and early stopping. An optional symbolic-regression step (PySR) fits a closed-form expression to the network's output.

## Results

- ~2.4× improvement in measurement accuracy compared with the area-based correction, across 2+ TB of data
- Reduced residual errors and long-tail outliers
- No generator-model dependence in the GNN training

## Data

- ALICE minimum-bias Pb–Pb collisions, 0–10% centrality
- PYTHIA pp Monte Carlo embedded in (anchored to) the minimum-bias Pb–Pb dataset, used for the dense-NN comparison and for evaluating performance against truth

ALICE data are not public, so the input files are not included in this repository.

## Repository structure

```
.
├── train_gnn_ics.py             # GNN: graph construction, training, evaluation, TorchScript export
├── neural_network_approach.py   # Dense NN trained on MC truth (comparison)
├── GNN_ICS.png                  # Results figure
└── README.md
```

## Running the code

```bash
pip install numpy pandas scipy scikit-learn matplotlib uproot awkward torch torch_geometric tensorflow
python train_gnn_ics.py
python neural_network_approach.py
```

Input file names are set at the top of each script.

**Tools:** PyTorch, PyTorch Geometric, TensorFlow/Keras, scikit-learn, SciPy, PySR, uproot, awkward
