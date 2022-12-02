# Copolymer Machine Learning
<a href="https://zenodo.org/badge/latestdoi/488046493"><img src="https://zenodo.org/badge/488046493.svg" alt="DOI"></a>

Code and data for the paper [Machine learning strategies for the structure-property relationship of copolymers](https://www.cell.com/iscience/fulltext/S2589-0042(22)00857-4).

A machine-learning (ML) implementation that incorporate the information of both **molecular composition** and **sequence distribution** of copolymers including **random, block, and gradient** copolymers. Please refer to our work "Machine Learning Strategies for the Structure-Property Relationship of Copolymers" for additional details.

<img src="Copolymers.png" width="60%">

## General Use
Given the molecular composition (SMILES of monomers) and copolymer sequence type (random, block, gradient), the ML model can incorporate both information and establish the structure-property relationship.
1. Datasets used in this work are uploaded to the `/datasets` folder.
2. Train model on each dataset, for example:
```
python train_dataset1.py --model 'RNN'
```
model options are 'CNN', 'FFNN', 'RNN', 'Fusion'
