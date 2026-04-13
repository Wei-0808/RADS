# RADS

Official code for **RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings**.

## Overview

RADS is an RL-based target sample selector for low-resource, class-imbalanced transfer learning.  
It combines:

1. source-domain active learning with MC dropout,
2. BALD-based informativeness scoring,
3. prior-aware utility for class-mixture control,
4. diversity-aware sequential selection with a dueling DQN sampler.

The selected target reports are then annotated and jointly used with the source data for transfer learning.

## Main experiments

This repository focuses on the two main transfer settings in the paper:

- CHIFIR → PIFIR
- MIMIC-CXR → PIFIR

## Repository structure

```text
RADS/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── configs/
├── scripts/
├── src/rads/
├── notebooks/
├── data/
├── outputs/
└── tests/
