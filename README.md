# RADS

This is the official implementation of **RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings**.

## Overview

RADS is an RL-based target sample selector for low-resource, class-imbalanced transfer learning.  
It combines:

1. source-domain active learning with MC dropout,
2. BALD-based informativeness scoring,
3. prior-aware utility for class-mixture control,
4. diversity-aware sequential selection with a dueling DQN sampler.

The selected target reports are then annotated and jointly used with the source data for transfer learning.

## Methodology

<p align="center">
  <img src="docs/method.png" alt="RADS methodology overview" width="360" />
</p>

At each step the active learner computes MC-dropout predictions on an unlabelled
target report `x_i`, yielding the state `s_i` (BALD score, log-probs, predictive
entropy, budget usage). The RL sampler agent takes action `A_i` — select or
skip — and receives a reward `R_i` that combines prior-aware utility with a
diversity / non-redundancy term. Selected reports are annotated and added to
the target-domain subset used jointly with the source data for transfer
learning.

## Main experiments

This repository focuses on the two main transfer settings in the paper:

- CHIFIR → PIFIR
- MIMIC-CXR → PIFIR


## Data Access

The original clinical datasets are not redistributed in this repository.

Please obtain them from the official sources:

PIFIR: https://physionet.org/content/pifir/1.0.0/

CHIFIR: https://physionet.org/content/corpus-fungal-infections/1.0.2/

MIMIC-CXR: https://physionet.org/content/mimic-cxr/2.1.0/

Place the files under data/raw/.

After download, the expected layout is:

```
data/raw/
├── CHIFIR_reports_dev.csv
├── CHIFIR_reports_test.csv
├── PIFIR_reports_dev.csv
├── PIFIR_reports_test.csv
├── MIMIC_reports_dev.csv     # MIMIC → PIFIR setting only
└── MIMIC_reports_test.csv
```

## Quickstart

Tested with Python 3.10 and a single CUDA GPU (CPU also works for the smoke test).

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Place clinical CSVs under data/raw/ (see Data Access above)

# 3. Run the CHIFIR → PIFIR pipeline
python -m scripts.run_chifir_to_pifir --config configs/chifir_to_pifir.yaml

# 4. Run the MIMIC-CXR → PIFIR pipeline
python -m scripts.run_mimic_to_pifir --config configs/mimic_to_pifir.yaml
```

Each script will:

1. fine-tune the source-domain classifier (ClinicalBERT by default) and save a
   checkpoint under `outputs/<experiment>/source_ckpt/best/`,
2. run MC-dropout to compute BALD features over the target pool,
3. train the dueling-DQN selector and write `selected_indices.json` plus a
   `selection_summary.json` to `outputs/<experiment>/`.

To reuse a previously trained source checkpoint and skip step (1):

```bash
python -m scripts.run_chifir_to_pifir \
    --config configs/chifir_to_pifir.yaml \
    --source-checkpoint outputs/chifir_to_pifir/source_ckpt/best
```

## Repository layout

```
configs/         YAML configs for each transfer setting
scripts/         CLI entry points (one per setting)
src/rads/        Library code: data, train, uncertainty, rl_selector, evaluate, metrics
tests/           Smoke tests (`pytest tests/`)
```

## Smoke test

```bash
pytest tests/
```


## Citation
